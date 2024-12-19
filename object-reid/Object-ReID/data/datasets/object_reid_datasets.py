# encoding: utf-8
import os.path as osp
from .bases import BaseImageDataset
import csv


class ObjectReIDDataset(BaseImageDataset):
    """
    Basis for ReID datasets for objects.
    Can technically be used as is but should be subclassed.

    Args:
    - dataset_dir (str): Dataset directory in root folder
    - name (str): Dataset name to display
    - only_eval (bool): Dataset does not have a train split
    - root (str): Root folder for all datasets
    - ds_type (str): Should be "object" or "object-masked" if segmentation mask to remove background is used
    - verbose (bool): Print out dataset statistics
    """
    def __init__(self, dataset_dir, name, only_eval=False, root='./toDataset', ds_type="object", verbose=True, **kwargs):
        super(ObjectReIDDataset, self).__init__(ds_type)
        self.dataset_dir = osp.join(root, dataset_dir)
        if not only_eval:
            self.train_split = osp.join(self.dataset_dir, 'train.csv')
        self.query_split = osp.join(self.dataset_dir, 'query.csv')
        self.gallery_split = osp.join(self.dataset_dir, 'test.csv')

        self._check_before_run(only_eval)

        # cam ids are all unique and quite meaningless; included here for compatability
        running_camid = 0
        if not only_eval:
            train, running_camid, classes = self._process_split(self.train_split, relabel=True, running_camid=running_camid)
            self.class_map = {e: i for i, e in enumerate(classes)}
        else:
            train = []
        query, running_camid, classes_ = self._process_split(self.query_split, relabel=False, running_camid=running_camid)
        gallery, running_camid, classes_ = self._process_split(self.gallery_split, relabel=False, running_camid=running_camid)

        if verbose:
            print(f"=> {name} loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        if len(self.train) > 0:
            self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_cls = self.get_imagedata_info(self.train)
        else:
            self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_cls = 0, 0, 0, 0
        if len(self.query) > 0:
            self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_cls = self.get_imagedata_info(self.query)
        else:
            self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_cls = 0, 0, 0, 0
        if len(self.gallery) > 0:
            self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_cls = self.get_imagedata_info(self.gallery)
        else:
            self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_cls = 0, 0, 0, 0

    def _check_before_run(self, only_eval):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if (not only_eval) and not osp.exists(self.train_split):
            raise RuntimeError("'{}' is not available".format(self.train_split))
        if not osp.exists(self.query_split):
            raise RuntimeError("'{}' is not available".format(self.query_split))
        if not osp.exists(self.gallery_split):
            raise RuntimeError("'{}' is not available".format(self.gallery_split))

    def _process_split(self, file_path, relabel=False, running_camid=0):
        """
        Load and prepare dataset part.

        Args:
        - file_path (str): Path to .csv file
        - relabel (bool): Create artificial continuous IDs instead of IDs in .csv
        - running_camid (int): Helper variable since camid is not really defined for OID
        """
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if self.ds_type == "object":
                assert header == ["path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"]
                img_data = [[r[0], r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5]), int(r[6]), int(r[7])] for r in reader]
                data_ind = {
                    "path": 0,
                    "class": 1,
                    "id": 2,
                    "img_num": 3,
                    "ymax": 4,
                    "ymin": 5,
                    "xmax": 6,
                    "xmin": 7
                }
            elif self.ds_type == "object-masked":
                assert header == ["path", "mask_path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"]
                img_data = [[r[0], r[1], r[2], int(r[3]), int(r[4]), int(r[5]), int(r[6]), int(r[7]), int(r[8])] for r in reader]
                data_ind = {
                    "path": 0,
                    "mask_path": 1,
                    "class": 2,
                    "id": 3,
                    "img_num": 4,
                    "ymax": 5,
                    "ymin": 6,
                    "xmax": 7,
                    "xmin": 8
                }
                mask_path_ind = data_ind["mask_path"]
            else:
                raise NotImplementedError

        obj_id_ind = data_ind["id"]
        img_path_ind = data_ind["path"]
        obj_class_ind = data_ind["class"]
        img_num_ind = data_ind["img_num"]
        ymax_ind = data_ind["ymax"]
        ymin_ind = data_ind["ymin"]
        xmax_ind = data_ind["xmax"]
        xmin_ind = data_ind["xmin"]

        oid_container = set()
        for e in img_data:
            oid_container.add(e[obj_id_ind])
        oid2label = {obj_id: label for label, obj_id in enumerate(oid_container)}

        dataset = []
        classes = set()
        for camid, e in enumerate(img_data):
            if relabel:
                obj_id = oid2label[e[obj_id_ind]]
            else:
                obj_id = e[obj_id_ind]
            ds_entry = (
                e[img_path_ind],
                obj_id,
                running_camid + camid,
                e[obj_class_ind],
                e[img_num_ind],
                e[ymax_ind],
                e[ymin_ind],
                e[xmax_ind],
                e[xmin_ind]
            )
            if self.ds_type == "object":
                dataset.append(ds_entry)
            elif self.ds_type == "object-masked":
                dataset.append((ds_entry[0], e[mask_path_ind]) + ds_entry[1:])
            classes.add(e[obj_class_ind])
        return dataset, running_camid + len(img_data), classes


class CO3D_ReID_v1(ObjectReIDDataset):
    """
    Initial experimental dataset created from CO3D dataset.
    Approximately 17 images per identity sampled randomly from video.
    Classes "banana", "kite", "parkingmeter" and "skateboard" only present in test split.
    Identities of remaining classes split 50/50 between train/test+query per class.
    Images per identity split 80/20 between test/query.
    Bounding Boxes created using only the segmentation masks.

    # identities:    18591
    # images train: 161255
    # images test:  134681
    # images query:  33910
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v1, self).__init__(
            "co3d_reid_v1",
            "CO3D_ReID_v1",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_Masked_v1(ObjectReIDDataset):
    """
    Same as CO3D_ReID_v1 but with background masked and set to [0, 0, 0].

    # identities:    18591
    # images train: 161255
    # images test:  134681
    # images query:  33910
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_Masked_v1, self).__init__(
            "co3d_reid_masked_v1",
            "CO3D_ReID_Masked_v1",
            root=root,
            ds_type="object-masked",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v2(ObjectReIDDataset):
    """
    Dataset created from CO3D_ReID_v1 dataset.
    Bounding Boxes were created using an object detector.
    The highest scored BB with an IOU > 0.25 with the GT from CO3D_ReID_v1 was chosen.
    Approximately 17 images per identity sampled randomly from video.
    Classes "banana", "kite", "parkingmeter" and "skateboard" only present in test split.
    Identities of remaining classes split 50/50 between train/test+query per class.
    Images per identity split 80/20 between test/query.

    # identities:    18590
    # images train: 161097
    # images test:  134547
    # images query:  33875
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v2, self).__init__(
            "co3d_reid_v2",
            "CO3D_ReID_v2",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v3(ObjectReIDDataset):
    """
    Dataset sampled from CO3D_ReID_v1 with approximately 10% of its IDs.
    Used for experiments with embedding dimension.

    # identities:    1797
    # images train: 15705
    # images test:  13097
    # images query:  3305
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v3, self).__init__(
            "co3d_reid_v3",
            "CO3D_ReID_v3",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v4(ObjectReIDDataset):
    """
    CO3D_ReID_v1 dataset without classes "car" and "stopsign" because they
    are either faulty or impossible to distinguish.

    # identities:    X
    # images train: X
    # images test:  X
    # images query:  X
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v4, self).__init__(
            "co3d_reid_v4",
            "CO3D_ReID_v4",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v5(ObjectReIDDataset):
    """
    CO3D_ReID_v1 dataset without classes "car", "stopsign", "broccoli", "chair",
    "microwave", "motorcycle", "parkingmeter" and "tv" because they are either
    faulty or hard to distinguish.

    # identities:    X
    # images train: X
    # images test:  X
    # images query:  X
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v5, self).__init__(
            "co3d_reid_v5",
            "CO3D_ReID_v5",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v6(ObjectReIDDataset):
    """
    CO3D_ReID_v5 dataset with as many samples as CO3D_ReID_v8.

    # identities:    X
    # images train: X
    # images test:  X
    # images query:  X
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v6, self).__init__(
            "co3d_reid_v6",
            "CO3D_ReID_v6",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v7(ObjectReIDDataset):
    """
    CO3D_ReID_v5 dataset with 75% of the classes and as many samples as CO3D_ReID_v8.

    # identities:    X
    # images train: X
    # images test:  X
    # images query:  X
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v7, self).__init__(
            "co3d_reid_v7",
            "CO3D_ReID_v7",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v8(ObjectReIDDataset):
    """
    CO3D_ReID_v5 dataset with 50% of the classes.

    # identities:    X
    # images train: X
    # images test:  X
    # images query:  X
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v8, self).__init__(
            "co3d_reid_v8",
            "CO3D_ReID_v8",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v10(ObjectReIDDataset):
    """
    Dataset created from CO3D dataset.
    Only images which are part of image sequences of length 20 or greater with
     consecutive bounding-box overlaps with an IoU larger than 0.0 were included.
    Approximately 19 images per identity sampled randomly from video.
    Classes "banana", "kite", "parkingmeter" and "skateboard" only present in test split.
    Identities of remaining classes split 50/50 between train/test+query per class.
    Images per identity split 80/20 between test/query.
    Bounding Boxes created using only the segmentation masks.

    # identities:    18591
    # images train: 173460
    # images test:  145136
    # images query:  36284
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v10, self).__init__(
            "co3d_reid_v10",
            "CO3D_ReID_v10",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v11(ObjectReIDDataset):
    """
    Like CO3D_ReID_v10 but without any test-only classes.

    # identities:    18591
    # images train: 177660
    # images test:  141776
    # images query:  35444
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v11, self).__init__(
            "co3d_reid_v11",
            "CO3D_ReID_v11",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v12(ObjectReIDDataset):
    """
    Dataset sampled from CO3D_ReID_v10 with approximately 10% of its IDs.

    # identities:    1734
    # images train: 16980
    # images test:  14160
    # images query:  3540
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v12, self).__init__(
            "co3d_reid_v12",
            "CO3D_ReID_v12",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )

class CO3D_ReID_v13(ObjectReIDDataset):
    """
    CO3D_ReID_v10 dataset with as many samples as CO3D_ReID_v15.

    # identities:    X
    # images train: X
    # images test:  X
    # images query:  X
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v13, self).__init__(
            "co3d_reid_v13",
            "CO3D_ReID_v13",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )

class CO3D_ReID_v14(ObjectReIDDataset):
    """
    CO3D_ReID_v10 dataset with 75% of the classes and as many samples as CO3D_ReID_v15.

    # identities:    X
    # images train: X
    # images test:  X
    # images query:  X
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v14, self).__init__(
            "co3d_reid_v14",
            "CO3D_ReID_v14",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class CO3D_ReID_v15(ObjectReIDDataset):
    """
    CO3D_ReID_v10 dataset with 50% of the classes.

    # identities:    X
    # images train: X
    # images test:  X
    # images query:  X
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(CO3D_ReID_v15, self).__init__(
            "co3d_reid_v15",
            "CO3D_ReID_v15",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )

class GoogleScan_ReID_v1(ObjectReIDDataset):
    """
    Dataset created from thumbnails of "Google Scanned Objects" dataset.
    Every class has only one example.
    All images have a pure (255, 255, 255) white background.
    Contains no train split and is only meant for evaluation.

    # identities:    1030
    # images train:     0
    # images test:   4120
    # images query:  1030
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(GoogleScan_ReID_v1, self).__init__(
            "gs_reid_v1",
            "GoogleScan_ReID_v1",
            only_eval=True,
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class Redwood_ReID_v1(ObjectReIDDataset):
    """
    Dataset created from "A Large Dataset of Object Scans" dataset, dubbed "Redwood" (by me).
    Bounding Boxes (BBs) were created using an object detector.
    The selected BB had the highest score out of all BBs fitting the following criteria:
    1. Intersect the center cell of a 3x3 grid overlayed over the image.
    2. Cover atleast 1/50 of the area of the image.
    Only 189 images in the entire dataset had no BBs that met these criteria and had to be discarded.
    The fact that there were no Ground Truth BBs and the object categories do not comform
     to any object detection dataset causes the quality of the BBs to be questionable!
    Exactly 20 images per identity sampled randomly from video.
    Identities of all classes split 50/50 between train/test+query per class.
    Images per identity split 80/20 between test/query.

    # identities:    10920
    # images train: 111045
    # images test:   85746
    # images query:  21440
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(Redwood_ReID_v1, self).__init__(
            "redwood_reid_v1",
            "Redwood_ReID_v1",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class Redwood_ReID_COCO(ObjectReIDDataset):
    """
    Dataset created from "A Large Dataset of Object Scans" dataset, dubbed "Redwood" (by me).
    Bounding Boxes (BBs) were created using an object detector.
    Only images containing classes which are also present in the COCO (and CO3D) dataset were considered!
    Whether this improves the quality of the BBs remains to be seen.
    The selected BB had the highest score out of all BBs fitting the following criteria:
    1. Intersect the center cell of a 3x3 grid overlayed over the image.
    2. Cover atleast 1/50 of the area of the image.
    Exactly 20 images per identity sampled randomly from video.
    The different splits are entirely contained within their respective splits from Redwood_ReID_v1.
    Identities of all classes split ~50/50 between train/test+query per class.
    Images per identity split ~80/20 between test/query.

    # identities:    3817
    # images train: 38451
    # images test:  30297
    # images query:  7574
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(Redwood_ReID_COCO, self).__init__(
            "redwood_reid_coco",
            "Redwood_ReID_COCO",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class Combined_Redwood_CO3D_v1(ObjectReIDDataset):
    """
    Dataset combined to be 50/50 from Redwood_ReID_v1 and CO3D_ReID_v1.
    Similar size as CO3D_ReID_v1.

    # identities:   15162
    # images train: 157659
    # images test:  131839
    # images query: 33101
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(Combined_Redwood_CO3D_v1, self).__init__(
            "combined_redwood_co3d_reid_v1",
            "Combined_Redwood_CO3D_v1",
            root=root,
            ds_type="object",
            verbose=verbose,
            kwargs=kwargs
        )


class OHO_ReID_v1(ObjectReIDDataset):
    """
    Dataset created from SONARO OHO datset.
    Contains images of objects recorded in front of a green screen.
    Background is masked out with [0, 0, 0].
    Contains only eval splits!

    # identities:     44
    # images train:    0
    # images test:   704
    # images query:  176
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(OHO_ReID_v1, self).__init__(
            "oho_reid_v1",
            "OHO_ReID_v1",
            only_eval=True,
            root=root,
            ds_type="object-masked",
            verbose=verbose,
            kwargs=kwargs
        )


class OHO_ReID_v2(ObjectReIDDataset):
    """
    Dataset created from OHO_ReID_v1.
    All non-tool classes were filtered out.
    Contains images of objects recorded in front of a green screen.
    Background is masked out with [0, 0, 0].
    Contains only eval splits!

    # identities:     27
    # images train:    0
    # images test:   432
    # images query:  108
    """
    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(OHO_ReID_v2, self).__init__(
            "oho_reid_v2",
            "OHO_ReID_v2",
            only_eval=True,
            root=root,
            ds_type="object-masked",
            verbose=verbose,
            kwargs=kwargs
        )
