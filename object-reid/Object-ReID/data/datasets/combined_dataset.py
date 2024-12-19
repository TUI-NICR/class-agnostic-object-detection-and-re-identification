from .bases import BaseImageDataset
import os.path as osp
import csv


class CombinedImageDataset(BaseImageDataset):
    """
    Image Dataset which is dynamically combined from multiple existing datasets.
    Only supports datasets of type "object".

    Args:
    - train_splits (list): List of dataset directory names whose train.csv should be used in training
    - query_splits (list): List of dataset directory names whose query.csv and test.csv should be used in evaluation
    - add_gallery_splits (list): List of dataset directory names whose test.csv should be appended to the galery split
    - name (str): Dataset name to display
    - root (str): Root folder for all datasets
    - verbose (bool): Print out dataset statistics
    """

    def __init__(self, train_splits, query_splits, add_gallery_splits=[], name="CombinedImageDataset", root='./toDataset', verbose=True):
        super(CombinedImageDataset, self).__init__("object")
        self.train_splits = []
        for ds in train_splits:
            ds = osp.join(root, ds, "train.csv")
            self.train_splits.append(ds)
        self.gallery_splits = []
        self.query_splits = []
        for ds in query_splits:
            ds_q = osp.join(root, ds, "query.csv")
            self.query_splits.append(ds_q)
            ds_g = osp.join(root, ds, "test.csv")
            self.gallery_splits.append(ds_g)
        self.add_gallery_splits = []
        for ds in add_gallery_splits:
            ds = osp.join(root, ds, "test.csv")
            self.add_gallery_splits.append(ds)

        self._check_before_run()

        running_camid = 0
        classes = set()
        trains = []
        for split in self.train_splits:
            train, running_camid, classes_ = self._process_split(split, relabel=True, running_camid=running_camid)
            classes = classes.union(classes_)
            trains.append(train)
        galleries = []
        for split in self.gallery_splits:
            gallery, running_camid, classes_ = self._process_split(split, relabel=False, running_camid=running_camid)
            classes = classes.union(classes_)
            galleries.append(gallery)
        queries = []
        for split in self.query_splits:
            query, running_camid, classes_ = self._process_split(split, relabel=False, running_camid=running_camid)
            classes = classes.union(classes_)
            queries.append(query)
        add_galleries = []
        for split in self.add_gallery_splits:
            gallery, running_camid, classes_ = self._process_split(split, relabel=False, running_camid=running_camid)
            classes = classes.union(classes_)
            add_galleries.append(gallery)

        self.class_map = {e: i for i, e in enumerate(classes)}

        train = self._join_splits([trains])[0]
        gallery, query = self._join_splits([galleries, queries])
        gallery = self._add_splits(gallery, add_galleries)

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

    def _check_before_run(self):
        for split in self.train_splits + self.gallery_splits + self.query_splits + self.add_gallery_splits:
            if not osp.exists(split):
                raise RuntimeError(f"'{split}' is not available")

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
            assert header == ["path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"]
            img_data = [[r[0], r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5]), int(r[6]), int(r[7])] for r in reader]

        oid_container = set()
        for img_path, obj_class, obj_id, img_num, ymax, ymin, xmax, xmin in img_data:
            oid_container.add(obj_id)
        oid2label = {obj_id: label for label, obj_id in enumerate(oid_container)}

        dataset = []
        classes = set()
        for camid, (img_path, obj_class, obj_id, img_num, ymax, ymin, xmax, xmin) in enumerate(img_data):
            if relabel:
                obj_id = oid2label[obj_id]
            dataset.append([img_path, obj_id, running_camid + camid, obj_class, img_num, ymax, ymin, xmax, xmin])
            classes.add(obj_class)

        return dataset, running_camid + len(img_data), classes

    def _join_splits(self, splits_list, oid_start=0):
        """
        Combine multiple loaded dataset parts.

        Args:
        - splits_list (list): List of list of processed splits
        - oid_start (int): Running OID to start from when relabeling OIDs
        """
        running_oid = oid_start
        oid_maps = [{} for _ in zip(*splits_list)]
        for splits, oid_map in zip(zip(*splits_list), oid_maps):
            for split in splits:
                for entry in split:
                    oid = entry[1]
                    if oid not in oid_map:
                        oid_map[oid] = running_oid
                        running_oid += 1
        joined = [[] for _ in splits_list]
        for splits, oid_map in zip(zip(*splits_list), oid_maps):
            for i, split in enumerate(splits):
                for entry in split:
                    entry[1] = oid_map[entry[1]]
                    joined[i].append(entry)
        return joined

    def _add_splits(self, exist_split, new_splits):
        """
        Add new dataset part to existing dataset part without changing OIDs of existing dataset part.

        Args:
        - exist_split (list): Processed split
        - new_splits (list): Processed split to add to exist_split
        """
        max_id = -1
        for entry in exist_split:
            oid = entry[1]
            if oid > max_id:
                max_id = oid
        new_split = self._join_splits([new_splits], oid_start=max_id+1)[0]
        exist_split += new_split
        return exist_split
