import sys
sys.path.append('.')

import os
import argparse
import os.path as osp
import json
import cv2
import numpy as np
import pickle as pkl

from time import time
from collections import defaultdict
from modeling import build_model as build_reid_model
from mmdet.apis import init_detector
from mmengine.config import Config

from caod.reid1 import build_reid_datasets_1, inference_reid_1, create_stat_features, apply_reid_1, reid_1_stats
from caod.detection import get_default_bbs, detection, det_stats
from caod.reid2 import build_reid_dataset_2, inference_reid_2, apply_reid_2, reid_2_stats


class LogStringParser(argparse.ArgumentParser):
    """
    Argumentparser which provides logs of all entered args.
    Set no_log=True in add_argument to avoid logging.
    """

    def __init__(self, *args, **kwargs):
        self._log_strings = []
        return super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        arg = args[0]
        no_log = kwargs.pop("no_log", False)
        if not no_log and "--" in arg:
            self._log_strings.append(arg.replace("--", ""))
        return super().add_argument(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args, namespace)
        logs = {}
        for arg in self._log_strings:
            value = getattr(args, arg)
            logs[arg] = value
        return args, logs


def parse_args():
    parser = LogStringParser('')
    parser.add_argument("--input_path", default="/path/to/paper/imgs_full/table", help="path to input gallery images", type=str)
    parser.add_argument("--input_gt_path", default="/path/to/paper/attach-benchmark/table_all_with_class.json", help="path to input gallery gt", type=str)
    parser.add_argument("--comparison_path", default="/path/to/paper/attach-benchmark/comparison_images_new", help="path to input query images", type=str)
    parser.add_argument("--comparison_gt_path", default="/path/to/paper/attach-benchmark/comparison_images.json", help="path to input query gt", type=str)
    parser.add_argument("--input_split", default=-1, type=int)

    parser.add_argument("--default_bb_size", default=512, help="size of boxes if not use_reid1", type=int)

    parser.add_argument("--use_reid1", action="store_true", default=True, help="Use ReID to filter boxes for detector.")
    parser.add_argument("--reid1_config_file", default="/path/to/paper/checkpoints/cp_regular/plain_resnet.yml", help="path to config file for first reid", type=str)
    parser.add_argument("--red", default=4, help="ratio to scale gallery images down by", type=int)
    parser.add_argument("--query_per_boxes", default=20, help="number of queries averaged for boxes. Int or 'all'.")
    parser.add_argument("--filter_quant", default=1.0, help="quantile of dists to keep", type=float)    # 0.05
    parser.add_argument("--threshold", default=0.5, help="normalized distance threshold", type=float)   # 0.2
    parser.add_argument("--use_stat_images", action="store_true", default=True, help="use additional images and calculate threshold based on stats")
    parser.add_argument("--stat_image_path", default="/path/to/paper/attach-benchmark/stat_images_attach.json", help="images for use_stat_images")
    parser.add_argument("--stat_image_quant", default=0.0, help="quantile of stat image dists to keep", type=float)
    parser.add_argument("--use_neg_dists", action="store_true", default=False, help="use neg_dists")
    parser.add_argument("--neg_dists", default="/does/not/exist", help="path to negative (background) distance maps", type=str)
    parser.add_argument("--augment", action="store_true", default=False, help="Use image augmentation on queries")
    parser.add_argument("--combine", action="store_true", default=True, help="Combine Bounding Boxes")
    parser.add_argument("--combine_p1", default=256, help="size for combined box if combine", type=int)
    parser.add_argument("--combine_p2", default=256, help="size for single box if combine", type=int)
    parser.add_argument("--combine_p3", default=0.8, help="IoU for further combination if combine", type=float)

    parser.add_argument("--combine_det_thr", default=0.6, help="IoU threshold for further combination of BBs before detection", type=float)
    parser.add_argument("--det_config_file", default="/path/to/paper/checkpoints/dino-4scale_r50_8xb2-12e_inference.py", help="path to config file for detection", type=str)
    parser.add_argument("--det_threshold", default=0.2, help="score threshold detection", type=float)  # 0.25
    parser.add_argument("--det_topk", default=300, help="top-k threshold detection per query", type=int)  # 300
    parser.add_argument("--det_resize", action="store_true", default=False, help="resize image patches to constant size")
    parser.add_argument("--det_size", default=600, help="resize size if det_resize", type=int)

    parser.add_argument("--combine_reid_thr", default=0.9, help="IoU threshold for further combination of BBs before ReID2", type=float)
    parser.add_argument("--reid2_config_file", default="/path/to/paper/checkpoints/cp_nl/CO3D_v1_baseline.yml", help="path to config file for first reid", type=str)
    parser.add_argument("--merge_reds", action="store_true", default=True, help="Merge boxes from reduced scales.")
    parser.add_argument("--merge_nms_thr", default=0.8, help="IoU threshold for NMS applied after merge.", type=float)
    parser.add_argument("--fuse_orders", action="store_true", default=True, help="Fuse rankings according to query_per_boxes.")
    parser.add_argument("--fuse_fn", default="min_dist", help="Function used to fuse rankings. Defined in reid2.py ORDER_FUSION.", type=str)

    parser.add_argument("--visualize_reid1", action="store_true", default=False, help="Visualize produced BBs of first ReiD", no_log=True)
    parser.add_argument("--visualize_det", action="store_true", default=False, help="Visualize produced BBs of detector", no_log=True)
    parser.add_argument("--visualize_reid2", action="store_true", default=False, help="Visualize produced ranking of first query of second ReiD", no_log=True)
    parser.add_argument("--vis_output_path", default="/path/to/results", help="path to visualization output", type=str, no_log=True)

    parser.add_argument("--stat_output_path", default="/path/to/results/stats", help="path to stats output", type=str, no_log=True)
    parser.add_argument("--reid1_stat_inter_threshold", default=0.8, help="area ratio cutoff for recall reid1", type=float)
    parser.add_argument("--det_stat_inter_threshold", default=0.8, help="area ratio cutoff for recall detection", type=float)
    parser.add_argument("--det_stat_union_threshold", default=1.4, help="area ratio cutoff for large box detection", type=float)
    return parser.parse_args()


def main():
    args, arg_logs = parse_args()
    if not args.query_per_boxes == "all":
        args.query_per_boxes = int(args.query_per_boxes)
        assert (not args.visualize_reid1) and (not args.visualize_det), "query_per_boxes != 'all' not compatible with visualization, too many images"
    all_stats = {}
    all_stats["args"] = arg_logs

    # setup results dir
    stat_output_path = args.stat_output_path
    if osp.isdir(stat_output_path):
        os.system(f"rm -r {stat_output_path}")
    os.mkdir(stat_output_path)

    from config import cfg
    cfg.merge_from_file(args.reid1_config_file)
    cfg.freeze()

    timing = defaultdict(float)
    start = time()
    input_path = args.input_path
    with open(args.input_gt_path, "r") as f:
        input_gts = json.load(f)

    if args.input_split >= 0:
        sp = args.input_split
        input_gts["images"] = input_gts["images"][sp*200:(sp+1)*200]

    # parse gallery inputs
    categories = {d["id"]: d["name"] for d in input_gts["categories"]}
    reverse_categories = {v: k for k, v in categories.items()}
    image_ids = {d["id"]: osp.join(input_path, d["file_name"]) for d in input_gts["images"]}
    image_anns = defaultdict(lambda: defaultdict(list))
    for d in input_gts["annotations"]:
        id_ = d["image_id"]
        if id_ in image_ids: # and d["category_id"] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            bbox = d["bbox"]
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            image_anns[image_ids[id_]][d["category_id"]].append((bbox, d["segmentation"], d["area"]))
    image_anns = dict(image_anns)

    with open(args.comparison_gt_path, "r") as f:
        comparison_gts = json.load(f)

    # # # # # # # # # # # # # # # #
    # Attach
    #comparison_gts = {k: v for k, v in comparison_gts.items() if k in [f"{x+y}.png" for x in range(0, 240, 20) for y in range(args.query_per_boxes)]}
    # # # # # # # # # # # # # # # #
    # Ikea
    #comparison_gts = {k: v for k, v in comparison_gts.items() if k in [f"{x+y}.png" for x in range(0, 300, 20) for y in range(0, 20, 20 // args.query_per_boxes)]}
    # # # # # # # # # # # # # # # #

    # parse query inputs
    reverse_comparison_gts = defaultdict(list)
    for k, v in comparison_gts.items():
        reverse_comparison_gts[int(v)].append(k)
    reverse_comparison_gts = dict(reverse_comparison_gts)

    query_groups = {}
    for class_, ps in reverse_comparison_gts.items():
        ps = sorted(ps)
        if args.query_per_boxes == "all": 
            query_groups[class_] = [tuple(ps)]
        else:
            assert len(ps) % args.query_per_boxes == 0, "Number of queries in class must be divisible by query_per_boxes"
            query_groups[class_] = [tuple(ps[i:i+args.query_per_boxes]) for i in range(0, len(ps), args.query_per_boxes)]

    all_gts = {
        "categories": categories,
        "reverse_categories": reverse_categories,
        "image_anns": image_anns,
        "query_gts": comparison_gts,
        "reverse_query_gts": reverse_comparison_gts,
        "query_groups" : query_groups
    }

    # load query images
    comparison_path = args.comparison_path
    comparison_images = []
    comparison_images_ = []
    for p, gt in comparison_gts.items():
        img = cv2.imread(osp.join(comparison_path, p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if args.augment:
            comparison_images.append([[img, cv2.flip(img, 1)], [int(gt), p]])
        else:
            comparison_images.append([img, [int(gt), p]])
        comparison_images_.append([img, [int(gt), p]])

    # load gallery images
    images = []
    images_det = {}
    for f in input_gts["images"]:
        p = osp.join(input_path, f["file_name"])
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_det[p] = img
        ims = [img]
        for r in range(2, args.red+1):
            img_ = cv2.resize(img, (img.shape[1]//r, img.shape[0]//r))
            ims.append(img_)
        images.append([ims, [p]])

    dataset_gallery, dataset_query = build_reid_datasets_1(cfg, images, comparison_images)
    stop = time()
    timing["prep_images_1"] = stop - start

    if args.use_reid1:
        start = time()
        reid_model = build_reid_model(cfg, num_classes=0)
        # cuda by default
        device = cfg.MODEL.DEVICE
        if 'cpu' not in device:
            reid_model.to(device=device)
        # load weights
        reid_model.load_param(cfg.TEST.WEIGHT)
        reid_model.eval()
        stop = time()
        timing["prep_reid_model1"] = stop - start

        # hook used to get last feature map of backbone
        intermediate = {}
        def hook(model, input, output):
            intermediate["e"] = output.detach()
        reid_model.base.register_forward_hook(hook)

        start = time()
        qf, qgts, qpaths, gallery = inference_reid_1(args, cfg, dataset_query, dataset_gallery, reid_model, intermediate)
        stop = time()
        timing["inference_reid_1"] = stop - start
        if args.use_stat_images:
            start = time()
            attach_features, avoid_boxes = create_stat_features(cfg, reid_model, intermediate, args.stat_image_path, args.red)
            stop = time()
            timing["inference_attach"] = stop - start
        else:
            attach_features = None
            avoid_boxes = None
        del reid_model

        reid_bbs, reid_dists = apply_reid_1(args, qf, qgts, qpaths, gallery, all_gts, compare_features=attach_features, avoid_boxes=avoid_boxes, timing=timing)
        all_stats["reid_1"] = reid_1_stats(args, reid_bbs, all_gts)
    else:
        reid_bbs = get_default_bbs(args, images_det, all_gts)
        reid_dists = {}

    start = time()
    config = Config.fromfile(args.det_config_file)
    det_model = init_detector(config, config.checkpoint, "coco", "cuda")
    stop = time()
    timing["prep_detector"] = stop - start

    det_bbs = detection(args, det_model, reid_bbs, images_det, all_gts, timing=timing)
    del det_model
    all_stats["boxes_det"] = det_stats(args, det_bbs, all_gts)

    del cfg
    from config import cfg
    cfg.merge_from_file(args.reid2_config_file)
    cfg.freeze()

    start = time()
    reid_model = build_reid_model(cfg, num_classes=0)
    # cuda by default
    device = cfg.MODEL.DEVICE
    if 'cpu' not in device:
        reid_model.to(device=device)
    # load weights
    reid_model.load_param(cfg.TEST.WEIGHT)
    reid_model.eval()
    stop = time()
    timing["prep_reid_model_2"] = stop - start

    start = time()
    dataset_cuts, dataset_query = build_reid_dataset_2(cfg, det_bbs, images_det, comparison_images_, args.combine_reid_thr)
    stop = time()
    timing["prep_images_2"] = stop - start

    start = time()
    gf, others, qf, qgts, qpaths = inference_reid_2(cfg, dataset_cuts, dataset_query, reid_model)
    del reid_model
    stop = time()
    timing["inference_reid_2"] = stop - start

    results = apply_reid_2(args, gf, others, qf, qpaths, all_gts, timing=timing)

    df, recalls, raw_stats = reid_2_stats(args, results, all_gts)

    with open(osp.join(stat_output_path, "stats.pkl"), "wb") as f:
        pkl.dump(raw_stats, f)

    all_stats["reid_2"] = recalls
    all_stats["timing"] = timing
    print(timing)

    # save stats
    df.to_csv(osp.join(stat_output_path, "recall.csv"))
    with open(osp.join(stat_output_path, "stats.json"), "w") as f:
        json.dump(all_stats, f)

    def _results(d):
        if isinstance(d, dict):
            return {_results(k): _results(v) for k, v in d.items()}
        elif isinstance(d, tuple):
            if all(isinstance(e, str) for e in d):
                return "_".join(d)
            else:
                return tuple(_results(e) for e in d)
        elif isinstance(d, list):
            return [_results(e) for e in d]
        elif isinstance(d, np.ndarray):
            return d.tolist()
        elif isinstance(d, (np.int32, np.int64)):
            return int(d)
        else:
            return d

    all_results = [reid_dists, reid_bbs, results]
    with open(osp.join(stat_output_path, "results.json"), "w") as f:
        json.dump(_results(all_results), f)


if __name__ == "__main__":
    main()
