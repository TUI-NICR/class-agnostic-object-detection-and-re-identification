import os
import cv2
import torch
import numpy as np
import os.path as osp

from mmcv.transforms import Compose
from time import time
from collections import defaultdict
from typing import Dict, Tuple, List
from argparse import Namespace
from numpy.typing import NDArray

from .tools import box_area, box_inter_union, overlap_box_groups


def get_default_bbs(
    args: Namespace,
    images_det: Dict[str, NDArray],
    all_gts: Dict
) -> Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], NDArray]]]]:
    """Create BB proposals using a sliding window approach instead of the
    first ReID stage.

    Args:
        args: Args of main.py.
        images_det: Galery images and their paths.
        all_gts: Dict containing all ground-truth values.

    Returns:
        Dict containing BB proposals in format (x, y, width, height, area).
        Grouped by Gallery path, class-code, resolution factor and Query-tuple.
    """
    size = args.default_bb_size
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    step = size // 2
    for p, img in images_det.items():
        values = []
        for y in range(0, img.shape[0], step):
            h = min(size, img.shape[0] - y)
            for x in range(0, img.shape[1], step):
                w = min(size, img.shape[1] - x)
                if w >= 256 and h >= 256:
                    values.append((x, y, w, h, w*h))
        for class_, queries in all_gts["query_groups"].items():
            for j in range(args.red):
                for query in queries:
                    results[p][class_][j][query] = np.array(values, dtype=np.int32)
    return results


def detection(
    args: Namespace,
    model: torch.nn.Module,
    reid_bbs: Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], NDArray]]]],
    images_det: Dict[str, NDArray],
    all_gts: Dict,
    timing: Dict ={}
) -> Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], Tuple[NDArray, NDArray, List[Tuple[int, int]]]]]]]:
    """Perform class-agnostic detection inside BB proposal areas for each image.

    Args:
        args: Args from main.py.
        model: MMDetection model.
        reid_bbs: BB proposals created by apply_reid_1 or get_default_bbs in format (x, y, width, height, area).
        images_det: Galery images and their paths.
        all_gts: Dict containing all ground-truth values.
        timing (optional): Dict to save timing information to. Modified inplace! Defaults to {}.

    Returns:
        Dict containing BBs in format (xmin, ymin, xmax, ymax), scores and coordinates relative to source image.
        Grouped by Gallery path, class-code, resolution factor and Query-tuple.
    """
    test_pipeline = model.cfg.test_pipeline
    if args.det_resize:
        for i, t in enumerate(test_pipeline):
            if t["type"] == "Resize":
                test_pipeline[i] = dict(type='Resize', scale=(args.det_size, args.det_size), keep_ratio=True)
    test_pipeline = Compose(test_pipeline)
    if args.visualize_det:
        vis_output_path_ = osp.join(args.vis_output_path, "VIS_D")
        if osp.isdir(vis_output_path_):
            os.system(f"rm -r {vis_output_path_}")
        os.mkdir(vis_output_path_)
    start = time()
    vis_time = 0
    dets = {}
    for path, classes in reid_bbs.items():
        img = images_det[path]
        img_crops = defaultdict(lambda: defaultdict(list))
        for class_, reds in classes.items():
            for red, queries in reds.items():
                for query, values in queries.items():
                    for x, y, w, h, a in values:
                        img_meta = ((x, y), class_, red, query)
                        # group images by size for batching
                        # group images by coords to avoid duplicates
                        if args.det_resize:
                            img_crops[(args.det_size, args.det_size)][(w, h, x, y)].append(img_meta)
                        else:
                            img_crops[(w, h)][(w, h, x, y)].append(img_meta)
        # combine very similar boxes to speed up computation
        if args.combine_det_thr < 1.0:
            bb_list = [((x, y, x+w, y+h), metas) for coords in img_crops.values() for (w, h, x, y), metas in coords.items()]
            n_grp, grp_labels = overlap_box_groups([bb for bb, _ in bb_list], args.combine_det_thr, True)
            bb_grps = [[] for _ in range(n_grp)]
            for i, label in enumerate(grp_labels):
                bb_grps[label].append(bb_list[i])
            img_crops = defaultdict(dict)
            for grp in bb_grps:
                boxes, metas = list(zip(*grp))
                boxes = np.array(boxes)
                # combined box is union of smaller ones
                x1, y1 = np.round(boxes[:, :2].min(axis=0))
                x2, y2 = np.round(boxes[:, 2:].max(axis=0))
                w = int(x2 - x1)
                h = int(y2 - y1)
                metas = [e for metas_ in metas for e in metas_]
                metas = [((int(x1), int(y1)), *meta[1:]) for meta in metas]
                img_ = img[y1:y2, x1:x2]
                if args.det_resize:
                    img_crops[(args.det_size, args.det_size)][(w, h, int(x1), int(y1))] = (img_, metas)
                else:
                    img_crops[(w, h)][(w, h, int(x1), int(y1))] = (img_, metas)
        else:
            for coords in img_crops.values():
                for (w, h, x, y), metas in coords.items():
                    img_ = img[y:y + h, x:x + w]
                    coords[(w, h, x, y)] = (img_, metas)
        batches = []
        batch_size = model.cfg.batch_size
        for coords in img_crops.values():
            batch = []
            for img in coords.values():
                batch.append(img)
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []
            if len(batch) > 0:
                batches.append(batch)
        results = []
        for batch in batches:
            inputs = []
            samples = []
            img_metas = []
            for img_, img_metas_ in batch:
                data = dict(img=img_, img_id=0)
                data = test_pipeline(data)
                inputs.append(data['inputs'])
                samples.append(data['data_samples'])
                img_metas.append(img_metas_)
            data = {'inputs': inputs, 'data_samples': samples}
            with torch.no_grad():
                result = model.test_step(data)
                results.append((result, img_metas))
        results = [(
            r.pred_instances.bboxes.detach().cpu().numpy().astype(np.int32),
            r.pred_instances.scores.detach().cpu().numpy(),
            img_meta
        ) for entry in results for r, img_metas in zip(*entry) for img_meta in img_metas]
        results_ = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for bboxes, scores, img_meta in results:
            coords, class_, red, query = img_meta
            # filter by threshold
            valids = np.logical_and(scores >= args.det_threshold,  ((bboxes[:, [2, 3]] - bboxes[:, [0, 1]]) > 0).all(axis=1))
            bboxes = bboxes[valids]
            scores = scores[valids]
            results_[class_][red][query].append((bboxes, scores, coords))
        for class_, reds in results_.items():
            for red, queries in reds.items():
                for query, e in queries.items():
                    bboxes, scores, coords = zip(*e)
                    coords = [coord for coord, bboxes_ in zip(coords, bboxes) for _ in bboxes_]
                    bboxes = np.concatenate(bboxes, axis=0)
                    scores = np.concatenate(scores, axis=0)
                    # filter by top-k
                    if len(scores) > args.det_topk * len(query):
                        inds = np.argsort(scores)[::-1][:args.det_topk * len(query)]
                        scores = scores[inds]
                        bboxes = bboxes[inds]
                        coords = [coords[i] for i in inds]
                    queries[query] = ((bboxes, scores, coords))
        results = results_
        dets[path] = dict(results)
        if args.visualize_det:
            start_vis = time()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for class_, reds in results.items():
                for red, res in reds.items():
                    res = list(res.values())[0]
                    img_ = img.copy()
                    for bbox, score, coords in zip(*res):
                        x, y = coords
                        xmin, ymin, xmax, ymax = bbox
                        cv2.rectangle(img_, (x + xmin, y + ymin), (x + xmax, y + ymax), (0, 0, 0), 3)
                    cv2.imwrite(osp.join(args.vis_output_path, "VIS_D", f"{osp.basename(path)}_{all_gts['categories'][class_]}_{red+1}.png"), img_)
            stop_vis = time()
            vis_time += stop_vis - start_vis
    stop = time()
    timing["inference_detection"] = stop - start - vis_time
    timing["vis_detection"] = vis_time
    return dets


def det_stats(
    args: Namespace,
    det_bbs: Dict[str, Dict[int, Dict[int, Dict[Tuple[str, ...], Tuple[NDArray, NDArray, List[Tuple[int, int]]]]]]],
    all_gts: Dict
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """Calculate some statistics for the detection stage.

    Args:
        args: Args of main.py.
        det_bbs: BBs, scores and coordinates output by detection.
        all_gts: Dict containing all ground-truth values.

    Returns:
       Dict containing stats grouped by Gallery path, class and resolution factor.
    """
    inter_thr = args.det_stat_inter_threshold
    union_thr = args.det_stat_union_threshold
    results = defaultdict(lambda: defaultdict(dict))
    for path, classes in det_bbs.items():
        for class_, reds in classes.items():
            anns = all_gts["image_anns"][path][class_]
            query_groups = all_gts["query_groups"][class_]
            for red, queries in reds.items():
                stats = {}
                stats["gt_boxes"] = len(anns)
                metric_names = ["recall", "empty", "large_box", "multiple"]
                for k in ["pred_boxes"] + metric_names:
                    stats[k] = 0
                num_query = len(query_groups)
                for query in query_groups:
                    res = queries.get(query, [])
                    res_ = list(zip(*res))
                    stats["pred_boxes"] += len(res_)
                    detected = np.zeros(len(anns))  # detection recall
                    empty = np.ones(len(res_))      # empty (wasted) boxes
                    large_box = np.zeros(len(res_)) # contains object but is much larger
                    multiple = np.zeros(len(res_))  # contains multiple objects
                    for i, (bbox1, seg, area) in enumerate(anns):
                        bbox1 = [int(v) for v in bbox1]
                        bbox1_area = box_area(bbox1)
                        for j, (bbox2, score, coords) in enumerate(res_):
                            x, y = coords
                            bbox2 = [bbox2[0]+x, bbox2[1]+y, bbox2[2]+x, bbox2[3]+y]
                            inter, union = box_inter_union(bbox1, bbox2)
                            if inter >= inter_thr * bbox1_area:
                                if empty[j] == 0:
                                    multiple[j] = 1
                                if union >= union_thr * bbox1_area:
                                    large_box[j] = 1
                                empty[j] = 0
                                detected[i] = 1
                    for k, m in zip(metric_names, [detected, empty, large_box, multiple]):
                        stats[k] += int(m.sum())
                for k in ["pred_boxes"] + metric_names: 
                    stats[k] /= num_query
                results[path][all_gts["categories"][class_]][red+1] = stats
    return dict(results)
