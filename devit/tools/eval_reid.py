import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import cv2
import numpy as np

from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from time import time
from tempfile import TemporaryDirectory

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup, default_argument_parser, launch
from detectron2.structures import BoxMode, Instances, pairwise_iou
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from tools.extract_instance_prototypes import main as extract_prototype_instances
from tools.run_sinkhorn_cluster import main as cluster_prototypes
from tools.sliding_window import SlidingWindowMapper, sliding_window


def make_gallery_dataset(filter_imgs=None):
    def return_func():
        P_GT = "/path/to/paper/attach-benchmark/table_all_with_class.json"
        with open(P_GT, "r") as f:
            input_gts = json.load(f)
        P = "/path/to/paper/attach-benchmark/input_images"
        dataset = []
        anns = defaultdict(list)
        for ann in input_gts["annotations"]:
            ann["bbox_mode"] = BoxMode.XYWH_ABS
            anns[ann["image_id"]].append(ann)
        anns = dict(anns)
        for img in sorted(input_gts["images"], key=lambda x: x["file_name"]):
            if filter_imgs and img not in filter_imgs:
                continue
            img["file_name"] = os.path.join(P, img["file_name"])
            img["image_id"] = img.pop("id")
            img["annotations"] = anns[img["image_id"]]
            dataset.append(img)
        return dataset
    return return_func


def make_query_dataset(filter_imgs=None):
    def return_func():
        P_GT = "/path/to/paper/attach-benchmark/comparison_images.json"
        with open(P_GT, "r") as f:
            comparison_gts = json.load(f)
        P = "/path/to/paper/attach-benchmark/comparison_images_new"
        dataset = []
        for img, ann in sorted(list(comparison_gts.items()), key=lambda x: x[0]):
            if filter_imgs and img not in filter_imgs:
                continue
            file_name = os.path.join(P, img)
            fp = Image.open(file_name)
            width, height = fp.size
            fp.close()
            d = {
                "file_name": file_name,
                "width": width,
                "height": height,
                "image_id": int(img.split(".")[0]),
                "annotations": [{
                    "bbox": [0, 0, width, height],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": int(ann)
                }]
            }
            dataset.append(d)
        return dataset
    return return_func


def make_query_dataset_with_masks(filter_imgs=None):
    def return_func():
        P = "/path/to/paper/queries"
        P_GT = "/path/to/paper/attach-benchmark/queries_detectron.json"
        with open(P_GT, "r") as f:
            comparison_gts = json.load(f)
        dataset = []
        for img in sorted(comparison_gts, key=lambda x: x["file_name"]):
            if filter_imgs and img["file_name"] not in filter_imgs:
                continue
            img["file_name"] = os.path.join(P, img["file_name"])
            dataset.append(img)
        return dataset
    return return_func


def main(args):
    VIS = False
    SLIDING_W = False
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    idx = args.opts.index("NUM_SHOTS")
    NUM_SHOTS = int(args.opts[idx+1])
    args.opts = args.opts[:idx] + args.opts[idx+2:]
    idx = args.opts.index("SPLIT")
    SPLIT = int(args.opts[idx+1])
    args.opts = args.opts[:idx] + args.opts[idx+2:]

    cfg.merge_from_list(args.opts)

    OFFSET = (SPLIT - 1) * NUM_SHOTS
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("shot-5", f"shot-{NUM_SHOTS}_split-{SPLIT}{'_sw' if SLIDING_W else ''}")
    print(f"Number of Shots: {NUM_SHOTS}")
    default_setup(cfg, args)
    TMP_OUT = TemporaryDirectory()  # tmp_outputs

    DatasetCatalog.register("full_gallery_dataset", make_gallery_dataset())
    #DatasetCatalog.register("full_query_dataset", make_query_dataset())
    DatasetCatalog.register("full_query_dataset", make_query_dataset_with_masks(
        [f"{x+y}.png" for x in range(0, 240, 20) for y in range(OFFSET, OFFSET + NUM_SHOTS)]
    ))
    with open("/path/to/paper/attach-benchmark/table_all_with_class.json", "r") as f:
        input_gts = json.load(f)
    classes = [x["name"] for x in sorted(input_gts["categories"], key=lambda y: y["id"])]

    MetadataCatalog.get("full_query_dataset").thing_classes = classes

    classes = classes + ["UNDEF"]
    colors = [tuple(cv2.cvtColor(np.array([int(i*180), 255, 255], np.uint8).reshape((1, 1, 3)), cv2.COLOR_HSV2BGR)[0, 0].tolist()) for i in range(len(classes))]
    MetadataCatalog.get("full_gallery_dataset").thing_classes = classes
    MetadataCatalog.get("full_gallery_dataset").thing_colors = colors

    t = cfg.MODEL.BACKBONE.TYPE[0]
    instance_path, p_time = extract_prototype_instances(
        model=f"vit{t}14",
        dataset="full_query_dataset",
        use_bbox="no",
        out_dir=TMP_OUT.name
    )
    prototype_path = cluster_prototypes(
        instance_path, num_prototypes=10,
        momentum=0.002, epochs=30,
        out_dir=TMP_OUT.name
    )

    #cfg.DE.CLASS_PROTOTYPES = prototype_path
    cfg.DE.CLASS_PROTOTYPES = cfg.DE.CLASS_PROTOTYPES.split(",")[0] + "," + prototype_path
    cfg.DATASETS.TRAIN = ("coco_plus_attach",)
    cfg.DATASETS.TEST = ("full_gallery_dataset",)
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(
        model,
        save_dir=cfg.OUTPUT_DIR
    ).resume_or_load(
        cfg.MODEL.WEIGHTS,
        resume=args.resume
    )
    model.eval()

    evaluator = COCOEvaluator(
        "full_gallery_dataset", output_dir=TMP_OUT.name, few_shot_mode=True,
        seen_cnames=[], unseen_cnames=classes, all_cnames=classes
    )

    if SLIDING_W:
        dataloader = build_detection_test_loader(cfg, sliding_window("full_gallery_dataset", 512), mapper=SlidingWindowMapper(cfg))
    else:
        dataloader = build_detection_test_loader(cfg, "full_gallery_dataset")
    results = []
    evaluator.reset()
    timing = 0

    with torch.no_grad():
        for inp in tqdm(dataloader):
            start = time()
            res = model(inp)
            res[0]["instances"].pred_classes -= 60
            res[0]["instances"].pred_classes[res[0]["instances"].pred_classes < 0] = 12
            results.append((inp[0], res[0]))
            stop = time()
            timing += stop - start
    torch.save(results, "tmp_outputs/results.pth")

    if SLIDING_W:
        eval_res = defaultdict(list)
        for inp, res in results:
            eval_res[inp["file_name"]].append((inp, res))
        results = []
        for file, entries in eval_res.items():
            insts = []
            for inp, outp in entries:
                inst = outp["instances"]
                inst.pred_boxes.tensor[:, [0, 2]] += inp["x_point"]
                inst.pred_boxes.tensor[:, [1, 3]] += inp["y_point"]
                inst._image_size = (1440, 2560)
                insts.append(inst)
            insts = Instances.cat(insts)
            outp = {"instances": insts}
            inp = {
                "image_id": entries[0][0]["image_id"] // 10_000,
                "file_name": file
            }
            results.append((inp, outp))
    for inp, res in results:
        evaluator.process([inp], [res])
    metrics = evaluator.evaluate()
    metrics["inf_time"] = timing
    metrics["prototype_time"] = p_time
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    """
    recalls = defaultdict(dict)
    for inp, res in results:
        for i, cls_ in enumerate(classes):
            gts = inp["instances"][inp["instances"].gt_classes == i]
            if len(gts) > 0:
                gt_boxes = gts.gt_boxes
                h, w = gts.image_size
                gt_boxes.scale(inp["width"] / w, inp["height"] / h)
                preds = res["instances"][res["instances"].pred_classes == i].to("cpu")
                pred_boxes = preds.pred_boxes
                ious = pairwise_iou(gt_boxes, pred_boxes)
                ious = ious >= 0.5
                found = ious.any(1)
                recall = float(torch.count_nonzero(found) / len(found))
                recalls[inp["file_name"]][cls_] = recall
            else:
                recalls[inp["file_name"]][cls_] = -1
    recalls = dict(recalls)
    with open("tmp_outputs/recalls.json", "w") as f:
        json.dump(recalls, f)
    """

    if VIS:
        for inp, res in results:
            for i, cls_ in enumerate(classes):
                inds = (res["instances"].pred_classes == i)
                part = res["instances"][inds]
                inds = torch.argsort(part.scores, descending=True)[:25]
                part = part[inds]
                vis = Visualizer(
                    cv2.cvtColor(cv2.imread(inp["file_name"]), cv2.COLOR_BGR2RGB),
                    MetadataCatalog.get("full_gallery_dataset")
                )
                img = vis.draw_instance_predictions(part.to("cpu"))
                img.save(f"tmp_outputs/{inp['image_id']}_{cls_}.png")


if __name__ == "__main__":
    sys.argv.insert(1, "--eval-only")
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
