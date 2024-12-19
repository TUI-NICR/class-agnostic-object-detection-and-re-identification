import os
import torch
import cv2
import json
import math
import torchvision.transforms as T
import os.path as osp
import numpy as np

from torch.utils.data import DataLoader
from collections import defaultdict
from time import time
from PIL import Image
from typing import Union, List, Tuple, Dict
from numpy.typing import NDArray
from yacs.config import CfgNode
from argparse import Namespace

from .dataset import ReIDDataset, collate_fn
from .components import combine_components
from .tools import box_area, box_inter_union


def build_reid_datasets_1(
    cfg: CfgNode,
    images: Union[List[Union[Image.Image, NDArray]], List[List[Union[Image.Image, NDArray]]]],
    comparison_images: Union[List[Union[Image.Image, NDArray]], List[List[Union[Image.Image, NDArray]]]]
):
    """Create datasets for first ReID stage from Query and Gallery images.

    Args:
        cfg: ReID Config.
        images: Gallery images in different resolutions.
        comparison_images: Query Images.

    Returns:
        Gallery and Query datasets.
    """
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    dataset_gallery = ReIDDataset(images, transform=transforms)
    transforms = T.Compose([
        T.Resize(cfg.INPUT.IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    dataset_query = ReIDDataset(comparison_images, transform=transforms)
    return dataset_gallery, dataset_query


def inference_reid_1(
    args: Namespace,
    cfg: CfgNode,
    dataset_query: ReIDDataset,
    dataset_gallery: ReIDDataset,
    reid_model: torch.nn.Module,
    intermediate: Dict
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[Tuple[torch.Tensor, List[str]]]]:
    """Creates features for first ReID stage.

    Args:
        args: Args from main.py.
        cfg: ReID Config.
        dataset_query: Query dataset.
        dataset_gallery: Gallery dataset.
        reid_model: ReID model.
        intermediate: Dict connected via hook to ReID model backbone output.

    Returns:
        Query features, Query ground-truths, Query image paths and Gallery features and image paths.
    """
    data_loader = DataLoader(dataset_query, batch_size=cfg.TEST.IMS_PER_BATCH, collate_fn=collate_fn, shuffle=False)
    queries = []
    qgts = []
    qpaths = []
    for batch in data_loader:
        data, others = batch
        gts, paths = others
        if args.augment:
            gts = [val for pair in zip(gts, gts) for val in pair]
            paths = [val for pair in zip(paths, paths) for val in pair]
        data = torch.stack(data, dim=0)
        with torch.no_grad():
            data = data.to(cfg.MODEL.DEVICE) if torch.cuda.device_count() >= 1 else data
            feat = reid_model(data)
            queries.append(feat)
            qgts += gts
            qpaths += paths
    qf = torch.cat(queries, dim=0)
    qf = torch.nn.functional.normalize(qf, dim=1, p=2)
    qgts = torch.tensor(qgts)

    data_loader = DataLoader(dataset_gallery, batch_size=cfg.TEST.IMS_PER_BATCH, collate_fn=collate_fn, shuffle=False)
    gallery = []
    for batch in data_loader:
        data, others = batch
        paths = others[0]
        batches = []
        for i in range(args.red):
            # batch per resolution
            d = data[i:][::args.red]
            batches.append(torch.stack(d, dim=0))
        results = []
        for data in batches:
            with torch.no_grad():
                data = data.to(cfg.MODEL.DEVICE) if torch.cuda.device_count() >= 1 else data
                reid_model(data)
                results.append([intermediate["e"], paths])
        gallery.append(results)
    gallery = [e for b in gallery for e in b]
    return qf, qgts, qpaths, gallery


def create_stat_features(
    cfg: CfgNode,
    reid_model: torch.nn.Module,
    intermediate: Dict,
    imgs_path: str,
    red: int
) -> Tuple[List[Tuple[torch.Tensor, List[str]]], Dict[str, List[Tuple[float, float, float, float]]]]:
    """Calculates features images to be used for dataset statistics in apply_reid_1.

    Args:
        cfg: ReID Config.
        reid_model: ReID model.
        intermediate: Dict connected via hook to ReID model backbone output.
        imgs_path: Path to json file with images.
        red: Max. resolution reduction factor.

    Returns:
        Features and image paths and Mapping from paths to object BBs in format (xmin, xmax, ymin, ymax) in [0.0, 1.0].
    """
    with open(imgs_path, "r") as f:
        imgs = json.load(f)
    images = []
    for c in imgs:
        img = cv2.imread(c)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ims = [img]
        for r in range(2, red+1):
            img_ = cv2.resize(img, (img.shape[1]//r, img.shape[0]//r))
            ims.append(img_)
        images.append([ims, [c]])

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    dataset = ReIDDataset(images, transform=transforms)
    data_loader = DataLoader(dataset, batch_size=cfg.TEST.IMS_PER_BATCH, collate_fn=collate_fn, shuffle=False)

    features = []
    for batch in data_loader:
        data, others = batch
        paths = others[0]
        batches = []
        for i in range(red):
            # batch per resolution
            d = data[i:][::red]
            batches.append(torch.stack(d, dim=0))
        results = []
        for data in batches:
            with torch.no_grad():
                data = data.to(cfg.MODEL.DEVICE) if torch.cuda.device_count() >= 1 else data
                reid_model(data)
                results.append([intermediate["e"], paths])
        features.append(results)
    features = [e for b in features for e in b]
    return features, imgs


def apply_reid_1(
    args: Namespace,
    qf: torch.Tensor,
    qgts: torch.Tensor,
    qpaths: List[str],
    gallery: List[Tuple[torch.Tensor, List[str]]],
    all_gts: Dict,
    compare_features: List[Tuple[torch.Tensor, List[str]]] =None,
    avoid_boxes: Dict[str, List[Tuple[float, float, float, float]]] =None,
    timing: Dict ={}
) -> Tuple[Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], NDArray]]]], Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], NDArray]]]]]:
    """Perform the first ReiD stage. Bounding Box proposals for the detection stage are created
    using the last feature map created by the reid_model backbone, connected component search over
    the distances between those features and the Query features and some post-processing.
    Distances can be filtered using datasets statistics without GT object BBs from image samples of
    source dataset.

    Args:
        args: Args of main.py.
        qf: Query features.
        qgts: Query ground-truths.
        qpaths: Query image paths.
        gallery: Gallery features and image paths.
        all_gts: Dict containing all ground-truth values.
        compare_features (optional): Features used to filter distances.
        avoid_boxes (optional): GT object BBs in format (xmin, xmax, ymin, ymax) in [0.0, 1.0] to avoid in compare_features.
        timing (optional): Dict to save timing information to. Modified inplace! Defaults to {}.

    Returns:
        Dict containing BB proposals in format (x, y, width, height, area), grouped by Gallery path, class-code, resolution factor
        and Query-tuple; Dict containing distance maps, grouped the same.
    """
    vis = args.visualize_reid1
    vis_output_path = args.vis_output_path
    if vis:
        for d in ["VIS_1", "CMPs"]:
            vis_output_path_ = osp.join(vis_output_path, d)
            if osp.isdir(vis_output_path_):
                os.system(f"rm -r {vis_output_path_}")
            os.mkdir(vis_output_path_)
    max_red = args.red
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    dist_returns = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for red in range(max_red):
        # one resolution per loop
        if compare_features is not None:
            # for each query calculate distances to reference dataset
            #  and use them to dynamically calculate a threshold ro ReID 
            vis_time = 0
            start = time()
            gc = compare_features[red:][::max_red]
            gc, paths = zip(*gc)
            gc = [e for t in gc for e in t]
            paths = [p for ps in paths for p in ps]
            c, h, w = gc[0].shape
            cmp_distmats = []
            for gf in gc:
                gf = gf.permute(1, 2, 0).view(h*w, c)
                gf = torch.nn.functional.normalize(gf, dim=1, p=2)
                m, n = qf.shape[0], gf.shape[0]
                distmat_ = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat_.addmm_(1, -2, qf, gf.t())
                distmat = distmat_.cpu().numpy()
                cmp_distmats.append(distmat)
            cmp_quants = []
            cmp_stats = []
            for k, path in enumerate(paths):
                dists = cmp_distmats[k]
                # image patches containing objects from queries
                # resulting features will be ignored
                boxes = avoid_boxes[path]
                num_q = dists.shape[0]
                keep = np.ones((num_q, h, w))
                for xmin, xmax, ymin, ymax in boxes:
                    xmin = math.floor(xmin * w)
                    xmax = math.ceil(xmax * w)
                    ymin = math.floor(ymin * h)
                    ymax = math.ceil(ymax * h)
                    keep[:, ymin:ymax, xmin:xmax] = 0
                    if vis and red == 1:
                        start_vis = time()
                        img = cv2.imread(path)
                        img[cv2.resize((1 - keep[0]), (img.shape[1], img.shape[0])).astype(np.bool_)] = (0, 0, 0)
                        cv2.imwrite(osp.join(vis_output_path, "CMPs", osp.basename(path)), img)
                        stop_vis = time()
                        vis_time += stop_vis - start_vis
                keep = keep.astype(np.bool_).reshape(dists.shape)
                dists = dists[keep].reshape((num_q, -1))
                if args.stat_image_quant == 0:
                    dists.sort()
                    quants = dists[:, 0]
                else:
                    quants = np.quantile(dists, args.stat_image_quant, 1)
                cmp_quants.append(quants)
                cmp_stats.append((dists.mean(1), dists.std(1), dists.max(1), dists.min(1)))
            cmp_quants = np.stack(cmp_quants).mean(axis=0)
            cmp_stats = np.stack(cmp_stats).mean(axis=0).T
            stop = time()
            timing["calc_stat_quants"] += stop - start - vis_time
            timing["vis_stat_images"] += vis_time
        # gallery samples for resolution
        g = gallery[red:][::max_red]
        g, paths = zip(*g)
        g = [e for t in g for e in t]
        paths = [p for ps in paths for p in ps]
        c, h, w = g[0].shape
        distmats = []
        start = time()
        for gf in g:
            gf = gf.permute(1, 2, 0).view(h*w, c)
            gf = torch.nn.functional.normalize(gf, dim=1, p=2)
            m, n = qf.shape[0], gf.shape[0]
            distmat_ = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat_.addmm_(1, -2, qf, gf.t())
            distmat = distmat_.cpu().numpy()
            distmats.append(distmat)
        # distances to queries
        distmats = np.array(distmats)
        stop = time()
        timing["calc_features_reid_1"] += stop - start
        for k, path in enumerate(paths):
            # for each gallery image
            base_name = osp.splitext(osp.basename(path))[0]
            if args.use_neg_dists:
                neg_dists = np.load(osp.join(args.neg_dists, base_name + ".npy"))
            if vis:
                start_vis = time()
                img = cv2.imread(path)
                stop_vis = time()
                timing["vis_images"] += stop - start
            img_ = Image.open(path)
            resolution = img_.size
            img_.close()
            for class_ in set(qgts.numpy()):
                class_ = int(class_)
                vis_time = 0
                start = time()
                distmats_ = distmats[:, qgts == class_]  # one class at a time
                qpaths_ = [p for i_, p in enumerate(qpaths) if i_ in np.arange(len(qpaths))[qgts == class_]]
                dists = distmats_[k]
                if vis:
                    start_vis = time()
                    min_dist = dists.min() / 2
                    mind_dist_p = np.unravel_index(dists.argmin(), (dists.shape[0], h, w))
                    resize_factors = tuple(s1 // s2 for s1, s2 in zip(img.shape, (h, w)))
                    mind_dist_bb = [(x * s, (x+1) * s) for x, s in zip(mind_dist_p[1:], resize_factors)]  # best matching point
                    stop_vis = time()
                    vis_time += stop_vis - start_vis
                max_diff = None
                avg_diff = None
                if compare_features is not None:
                    quants = cmp_quants[qgts == class_]
                    means, stds, maxs, mins = cmp_stats[qgts == class_].T
                    # calculate average difference between distances lower than their corresponding
                    #  quantile and said quantile
                    diffs = (quants - np.mean(dists, axis=1, where=(dists <= quants[:, np.newaxis]))) / (maxs - mins)
                    diffs[np.isnan(diffs)] = 0
                    # set distances larger than their corresponding quantile to the max value of that query
                    dists = np.where(dists > quants[:, np.newaxis], dists.max(axis=1, keepdims=True), dists)
                    max_diff = max(diffs)
                    avg_diff = np.array(diffs).mean()
                dists = dists.reshape(dists.shape[0], h, w)
                if args.query_per_boxes == "all":
                    dists = dists.sum(axis=0, keepdims=True)  # sum over all queries
                else:
                    dists_ = []
                    for group in all_gts["query_groups"][class_]:
                        inds = [i for i, e in enumerate(qpaths_) if e in group]
                        dists_.append(dists[inds].sum(axis=0, keepdims=True))
                    dists = np.concatenate(dists_, axis=0)
                norm = dists.max(axis=(1, 2), keepdims=True) - dists.min(axis=(1, 2), keepdims=True)
                dists = (dists - dists.min(axis=(1, 2), keepdims=True)) / norm  # normalize distances
                dists[np.isnan(dists)] = 1
                if args.use_neg_dists:
                    neg_dists = cv2.resize(neg_dists, (w, h))
                    dists = dists + 1 - neg_dists[np.newaxis]
                    dists = np.clip(dists, 0, 1)
                # filter positions via threshold and quantile
                quant = np.quantile(dists, args.filter_quant, 1, keepdims=True)
                dists[dists > quant] = 1
                dists[dists > args.threshold] = 1
                for query, d_ in zip(all_gts["query_groups"][class_], dists):
                    dist_returns[path][class_][red][query] = d_
                dists = np.stack([cv2.resize(d, resolution) for d in dists], axis=0)
                # binary mask to get connected components
                threshold = 1 - dists
                threshold[threshold > 0] = 1
                threshold = (threshold*255).astype(np.uint8)
                stop = time()
                timing["calc_dists_reid_1"] += stop - start - vis_time
                timing["vis_images"] += vis_time
                for query, t in zip(all_gts["query_groups"][class_], threshold):
                    start = time()
                    analysis = cv2.connectedComponentsWithStats(t, 4, cv2.CV_32S)
                    stop = time()
                    timing["connected_components"] += stop - start
                    values = analysis[2]  # bounding boxes of connected components
                    # first entry is entire image for some reason (background component?)
                    if values[0, 2] >= resolution[0] and values[0, 3] >= resolution[1]:
                        values = values[1:]
                    if args.combine:
                        V1 = args.combine_p1  # size for combined box
                        V2 = args.combine_p2  # size for single box
                        V3 = args.combine_p3  # IoU for combination
                        # combine small boxes if they are close enough to each other
                        #  and enlarge others (detector only works on larger crops)
                        start = time()
                        if len(values) > 0:
                            values = combine_components(values, V1, V2, V3, resolution)
                        stop = time()
                        timing["combine_components"] += stop - start
                    results[path][class_][red][query] = values
                    if vis:
                        dists = dists[0]
                        start = time()
                        dists = np.repeat(dists[..., np.newaxis], 3, 2)
                        mask = np.ones_like(dists, np.uint8)
                        # sample HSV color space for different color for each class
                        mask[..., 0] = mask[..., 0] * 180 // qgts.max().numpy() * class_
                        mask[..., 1:] = mask[..., 1:] * 255
                        mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
                        res = (dists * img + (1-dists) * mask).astype(np.uint8)
                        #res[mind_dist_bb[0][0]:mind_dist_bb[0][1], mind_dist_bb[1][0]:mind_dist_bb[1][1]]
                        cv2.rectangle(res, (mind_dist_bb[1][0], mind_dist_bb[0][0]), (mind_dist_bb[1][1], mind_dist_bb[0][1]), tuple(int(v) for v in 255 - mask[0, 0]), 3)
                        for x, y, w_, h_, a in values:
                            cv2.rectangle(res, (x, y), (x + w_, y + h_), (0, 0, 0), 3)
                        cv2.imwrite(osp.join(vis_output_path, "VIS_1", f"{base_name}_{all_gts['categories'][class_]}_{red+1}_{min_dist:.3f}_{avg_diff:.3f}-{max_diff:.3f}.png"), res)
                        stop = time()
                        timing["vis_images"] += stop - start
    return dict(results), dict(dist_returns)


def reid_1_stats(
    args: Namespace,
    reid_bbs: Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], NDArray]]]],
    all_gts: Dict
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """Calculate some statistics for first ReID stage.

    Args:
        args: Args of main.py.
        reid_bbs: BBs output by apply_reid_1.
        all_gts: Dict containing all ground-truth values.

    Returns:
        Dict containing stats grouped by Gallery path, class and resolution factor.
    """
    results = defaultdict(lambda: defaultdict(dict))
    for path, classes in reid_bbs.items():
        for class_, reds in classes.items():
            anns = all_gts["image_anns"][path][class_]
            for red, queries in reds.items():
                stats = {}
                stats["gt_boxes"] = len(anns) # ground truth boxes 
                stats["pred_boxes"] = 0       # produced boxes
                stats["recall"] = 0           # recall
                stats["empty"] = 0            # empty boxes
                for query, values in queries.items():
                    stats["pred_boxes"] += len(values)
                    empty = np.ones(len(values))
                    detected = np.zeros(len(anns))
                    for i, (bbox, seg, area) in enumerate(anns):
                        bbox = [int(v) for v in bbox]
                        bbox_area = box_area(bbox)
                        for j, (x, y, w, h, a) in enumerate(values):
                            inter, union = box_inter_union(bbox, [x, y, x+w, y+h])
                            if inter >= args.reid1_stat_inter_threshold * bbox_area:
                                detected[i] = 1
                                empty[j] = 0
                    stats["recall"] += int(detected.sum())
                    stats["empty"] += int(empty.sum())
                stats["pred_boxes"] /= len(queries)
                stats["recall"] /= len(queries)
                stats["empty"] /= len(queries)
                results[path][all_gts["categories"][class_]][red+1] = stats
    return dict(results)
