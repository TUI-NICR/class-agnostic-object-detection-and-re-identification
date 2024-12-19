import os
import cv2
import torch
import torchvision.transforms as T
import os.path as osp
import numpy as np
import pandas as pd
import pickle

from time import time
from torch.utils.data import DataLoader
from collections import defaultdict
from torchvision.ops import box_iou, nms
from typing import List, Dict, Union, Tuple, Any, Callable
from yacs.config import CfgNode
from numpy.typing import NDArray
from PIL.Image import Image
from argparse import Namespace

from .dataset import ReIDDataset, collate_fn
from .tools import overlap_box_groups


def build_reid_dataset_2(
    cfg: CfgNode,
    det_bbs: Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], Tuple[NDArray, NDArray, List[Tuple[int, int]]]]]]],
    images_det: Dict[str, NDArray],
    comparison_images: Union[List[Union[Image, NDArray]], List[List[Union[Image, NDArray]]]],
    combine_reid_thr: float
):
    """Create datasets for second ReID stage from Query images and BB cutouts of Gallery images.

    Args:
        cfg: ReID Config.
        det_bbs: BBs in format (xmin, ymin, xmax, ymax), scores and coordinates output by detection.
        images_det: Gallery images and their paths.
        comparison_images: Query Images.
        combine_reid_thr: IoU threshold for further combination of BBs.

    Returns:
        Dataset from Gallery images and Query dataset.
    """
    cut_images = []
    for path, classes in det_bbs.items():
        img = images_det[path]
        cut_images_ = defaultdict(list)
        for class_, reds in classes.items():
            for red, queries in reds.items():
                for query, res in queries.items():
                    res_ = zip(*res)
                    for bbox, score, coords in res_:
                        x, y = coords
                        xmin, ymin, xmax, ymax = bbox
                        cut_images_[(ymin+y, ymax+y, xmin+x, xmax+x)].append([class_, path, red, query, [xmin+x, ymin+y, xmax+x, ymax+y], score])
        if combine_reid_thr < 1.0:
            bb_list = [([bb[2], bb[0], bb[3], bb[1]], metas) for bb, metas in cut_images_.items()]
            n_grp, grp_labels = overlap_box_groups([bb for bb, _ in bb_list], combine_reid_thr, True)
            bb_grps = [[] for _ in range(n_grp)]
            for i, label in enumerate(grp_labels):
                bb_grps[label].append(bb_list[i])
            cut_images_ = {}
            for grp in bb_grps:
                boxes, metas = list(zip(*grp))
                boxes = np.array(boxes)
                # combined box is union of smaller ones
                x1, y1 = np.round(boxes[:, :2].min(axis=0))
                x2, y2 = np.round(boxes[:, 2:].max(axis=0))
                metas = [e for metas_ in metas for e in metas_]
                metas = [meta[:4] + [[x1, y1, x2, y2]] + meta[5:] for meta in metas]
                cut_images_[(y1, y2, x1, x2)] = metas
        cut_images += [[img[y1:y2, x1:x2], [metas]] for (y1, y2, x1, x2), metas in cut_images_.items()]
    transforms = T.Compose([
        T.Resize(cfg.INPUT.IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    dataset_cuts = ReIDDataset(cut_images, transform=transforms)
    dataset_query = ReIDDataset(comparison_images, transform=transforms)
    return dataset_cuts, dataset_query


def inference_reid_2(
    cfg: CfgNode,
    dataset_cuts: ReIDDataset,
    dataset_query: ReIDDataset,
    reid_model: torch.nn.Module
) -> Tuple[torch.Tensor, List[List[Tuple[Any, ...]]], torch.Tensor, torch.Tensor, List[str]]:
    """Creates features for second ReID stage.

    Args:
        cfg: ReID Config.
        dataset_cuts: Dataset from Gallery images.
        dataset_query: Query dataset.
        reid_model: ReID model.

    Returns:
        Gallery features, Gallery meta infos, Query features, Query ground-truths and Query image paths.
    """
    data_loader = DataLoader(dataset_cuts, batch_size=cfg.TEST.IMS_PER_BATCH, collate_fn=collate_fn, shuffle=False)
    gallery = []
    others = []
    for batch in data_loader:
        data, others_ = batch
        others += list(zip(*others_))
        data = torch.stack(data, dim=0)
        with torch.no_grad():
            data = data.to(cfg.MODEL.DEVICE) if torch.cuda.device_count() >= 1 else data
            feat = reid_model(data)
            gallery.append(feat)
    gf = torch.cat(gallery, dim=0)
    gf = torch.nn.functional.normalize(gf, dim=1, p=2)

    data_loader = DataLoader(dataset_query, batch_size=cfg.TEST.IMS_PER_BATCH, collate_fn=collate_fn, shuffle=False)
    queries = []
    qgts = []
    qpaths = []
    for batch in data_loader:
        data, others_ = batch
        gts, paths = others_
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
    return gf, others, qf, qgts, qpaths


ORDER_FUSION: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "min_dist": lambda distmat: torch.argsort(torch.min(distmat, dim=0, keepdim=True).values, dim=1),
    "avg_dist": lambda distmat: torch.argsort(torch.mean(distmat, dim=0, keepdim=True), dim=1)
}


def apply_reid_2(
    args: Namespace,
    gf: torch.Tensor,
    others: List[List[Tuple[int, str, int, Tuple[str, ...], Tuple[int, int, int, int], int]]],
    qf: torch.Tensor,
    qpaths: List[str],
    all_gts: Dict,
    timing: Dict ={}
) -> Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], Tuple[List[Tuple[int, int, int, int]], NDArray, NDArray]]]]]:
    """Perform the second ReiD stage. For each Query image rank Gallery image cutouts by their similarity.

    Args:
        args: Args of main.py.
        gf: Gallery features.
        others: Gallery meta infos.
        qf: Query features.
        qpaths: Query image paths.
        all_gts: Dict containing all ground-truth values.
        timing (optional): Dict to save timing information to. Modified inplace! Defaults to {}.

    Returns:
        Dict containing cutout BBs in format (xmin, ymin, xmax, ymax) and rankings.
        Grouped by Gallery path, class-code, resolution factor and Query-tuple.
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    #TMP_DISTS = []
    for i, entry in enumerate(others):
        for (gt, path, red, query, bbox, score) in entry[0]:
            results[path][gt][red][query].append((i, bbox, score))
    results = dict(results)
    if args.visualize_reid2:
        vis_output_path_ = osp.join(args.vis_output_path, "VIS_2")
        if osp.isdir(vis_output_path_):
            os.system(f"rm -r {vis_output_path_}")
        os.mkdir(vis_output_path_)
    for path, classes in results.items():
        if args.visualize_reid2:
            img = cv2.imread(path)
        start = time()
        for class_, reds in classes.items():
            if args.merge_reds:
                # merge boxes from all resolutions and perform NMS
                queries_ = defaultdict(list)
                for red, queries in reds.items():
                    for query, e in queries.items():
                        queries_[query] += e
                for query, e in queries_.items():
                    inds, bboxes, scores = zip(*e)
                    keep = set(nms(torch.tensor(bboxes, dtype=torch.float32), torch.tensor(scores), args.merge_nms_thr).tolist())
                    queries_[query] = [x for i, x in enumerate(e) if i in keep]
                reds_ = {-2: dict(queries_)}
            else:
                reds_ = reds
            res = defaultdict(dict)
            for red, queries in reds_.items():
                for query, e in queries.items():
                    qinds = torch.tensor([i for i, p in enumerate(qpaths) if p in query])
                    qf_ = qf[qinds]
                    m = qf_.shape[0]
                    inds, bboxes, scores = zip(*e)
                    gf_ = gf[torch.tensor(inds)]
                    n = gf_.shape[0]
                    distmat = torch.pow(qf_, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf_, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                    distmat.addmm_(1, -2, qf_, gf_.t())
                    #TMP_DISTS.append((distmat.cpu().numpy(), bboxes, path, class_, query))
                    all_orders = torch.argsort(distmat, dim=1).cpu().numpy()
                    if args.fuse_orders:
                        orders = ORDER_FUSION[args.fuse_fn](distmat).cpu().numpy()
                    else:
                        orders = all_orders
                    res[red][query] = (bboxes, orders, distmat.cpu().numpy())  # [N, 4], [M, N] in [0, N) or [1, M] in [0, N), [M, N] in [0, N)
                    if args.visualize_reid2:
                        start_vis = time()
                        img_ = img.copy()
                        # only first query and first 10 boxes visualized
                        for order, idx in enumerate(orders[0][:10]):
                            alpha = order / 10
                            color = tuple(cv2.cvtColor(np.array([int(alpha*180), 255, 255], np.uint8).reshape((1, 1, 3)), cv2.COLOR_HSV2BGR)[0, 0].tolist())
                            bbox = bboxes[idx]
                            xmin, ymin, xmax, ymax = bbox
                            width = xmax - xmin
                            height = ymax - ymin
                            #rec = img_[ymin:ymax, xmin:xmax]
                            #rec_ = rec.copy()
                            #cv2.rectangle(rec_, (0, 0), (width, height), color, 5)
                            #cv2.putText(rec_, str(order+1), (width//2, height//2), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
                            #rec = cv2.addWeighted(rec_, 1-alpha, rec, alpha, 0)
                            #img_[ymin:ymax, xmin:xmax] = rec
                            cv2.rectangle(img_, (xmin, ymin), (xmax, ymax), color, 3)
                            cv2.putText(img_, str(order+1), (xmin+width//2, ymin+height//2), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
                        cv2.imwrite(osp.join(args.vis_output_path, "VIS_2", f"{osp.basename(path)}_{all_gts['categories'][class_]}_{red+1}_{query[0].split('.')[0]}.png"), img_)
                        stop_vis = time()
                        timing["vis_reid_2"] += stop_vis - start_vis
            results[path][class_] = dict(res)
        stop = time()
        timing["calc_features_reid_2"] += stop - start
    timing["calc_features_reid_2"] -= timing["vis_reid_2"]
    #with open(args.stat_output_path + "/distmats.pkl", "wb") as f:
    #    pickle.dump(TMP_DISTS, f)
    return results


def _append_default_stats_inplace(data, index, path, categories, classes, all_reds, query_groups):
    if not isinstance(classes, dict):
        classes = [classes]
    if not isinstance(all_reds, list):
        all_reds = [all_reds]
    if not isinstance(query_groups, dict):
        query_groups = {classes[0]: [query_groups]}
    for class_ in classes:
        for red in all_reds:
            for query in query_groups[class_]:
                m = len(query)
                data.append(np.stack([np.repeat(np.nan, m), np.repeat(0, m), np.repeat(0, m), np.repeat(0, m), np.repeat(0, m), np.repeat(np.nan, m)], axis=-1))
                idx = [(path, categories[class_], red+1, q) for q in query]
                index += idx


def reid_2_stats(
    args: Namespace,
    results: Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], Tuple[List[Tuple[int, int, int, int]], NDArray, NDArray]]]]],
    all_gts: Dict
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Dict[str, float]]]]:
    """Calculate statistics for second ReID stage.
    These represent the final metrics used to evaluate the whole pipeline.

    Args:
        args: Args of main.py.
        results: BBs and rankings output by apply_reid_2.
        all_gts: Dict containing all ground-truth values.

    Returns:
        Dataframe containing all results and Dict containing stats grouped by class and resolution factor.
    """
    categories = all_gts["categories"]
    query_groups = all_gts["query_groups"]
    index = []
    data = []
    raw_stats = []
    if args.merge_reds:
        all_reds = [-2]
    else:
        all_reds = list(range(args.max_red))
    for path, classes in all_gts["image_anns"].items():
        reds_ = results.get(path, {})
        if len(reds_) == 0:
            _append_default_stats_inplace(data, index, path, categories, classes, all_reds, query_groups)
            for class_, anns in classes.items():
                bboxes1, _, _ = zip(*anns)
                bboxes1 = torch.tensor(bboxes1)
                for red in all_reds:
                    raw_stats.append((path, categories[class_], red+1, bboxes1, None, None, None))
            print(f"No predictions for {path}")
            continue
        for class_, anns in classes.items():
            if len(anns) == 0:
                continue
            bboxes1, _, _ = zip(*anns)
            bboxes1 = torch.tensor(bboxes1)
            g = bboxes1.shape[0]
            reds = reds_.get(class_)
            if reds is None:
                _append_default_stats_inplace(data, index, path, categories, class_, all_reds, query_groups)
                for red in all_reds:
                    raw_stats.append((path, categories[class_], red+1, bboxes1, None, None, None))
                continue
            for red in all_reds:
                queries = reds.get(red)
                if queries is None:
                    _append_default_stats_inplace(data, index, path, categories, class_, red, query_groups)
                    raw_stats.append((path, categories[class_], red+1, bboxes1, None, None, None))
                    continue
                for query in query_groups[class_]:
                    e = queries.get(query)
                    if e is None:
                        _append_default_stats_inplace(data, index, path, categories, class_, red, query)
                        raw_stats.append((path, categories[class_], red+1, bboxes1, query, None, None))
                        continue
                    bboxes2, orders, distmat = e
                    bboxes2 = torch.tensor(bboxes2)
                    raw_stats.append((path, categories[class_], red+1, bboxes1, query, bboxes2, distmat))
                    if args.fuse_orders:
                        orders = np.tile(orders, (len(query), 1))
                    m, n = orders.shape
                    idx = [(path, categories[class_], red+1, q) for q in query]
                    index += idx
                    ious = box_iou(bboxes1, bboxes2).numpy()  # [G, N]
                    valids = ious >= 0.5
                    recalled = valids.any(axis=1)   # [G]
                    base_recall = recalled.sum() / g
                    hits = valids.any(axis=0)       # [N]
                    base_precision = hits.sum() / n
                    if base_recall > 0:
                        matches = hits[orders]              # [M, N]
                        num_rel = matches.sum(axis=1)       # [M]
                        tmp_cmc = matches.cumsum(axis=1)    # [M, N]
                        CMC1 = (tmp_cmc[:, 0] > 0).astype(np.int32)  # [M]
                        CMC5 = (tmp_cmc[:, :5] > 0).any(axis=1).astype(np.int32)  # [M]
                        tmp_cmc = tmp_cmc / np.tile(np.arange(tmp_cmc.shape[1]) + 1.0, (tmp_cmc.shape[0], 1))  # [M, N]
                        tmp_cmc = tmp_cmc * matches         # [M, N]
                        AP = tmp_cmc.sum(axis=1) / num_rel  # [M]
                        AP_D = AP * base_recall
                        data.append(np.stack([AP, CMC1, CMC5, np.repeat(base_recall, m), AP_D, np.repeat(base_precision, m)], axis=-1))
                    else:
                        data.append(np.stack([np.repeat(np.nan, m), np.repeat(0, m), np.repeat(0, m), np.repeat(0, m), np.repeat(0, m), np.repeat(0, m)], axis=-1))
    index = pd.MultiIndex.from_tuples(index, names=["path", "class", "red", "query"])
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data, index, ["AP@recall", "CMC1", "CMC5", "det_recall", "AP", "det_precision"])
    recalls = defaultdict(dict)
    for class_, df_class in df.groupby("class"):
        for red, df_red in df_class.groupby("red"):
            mean = df_red.mean(skipna=True)
            recalls[class_][red] = {"mean_" + k: v for k, v in mean.items()}
    return df, dict(recalls), raw_stats
