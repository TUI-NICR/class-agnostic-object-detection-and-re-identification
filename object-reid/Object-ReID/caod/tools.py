import numpy as np
import torch
from torchvision.ops import box_iou as t_box_iou
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from typing import Tuple, Union, List
from numpy.typing import NDArray


def box_area(box: Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int]]) -> Union[float, int]:
    """Calculate Box area.

    Args:
        box: Box in format (xmin, ymin, xmax, ymax).

    Returns:
      Box area.
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def box_inter_union(
    box1: Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int]],
    box2: Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int]]
) -> Tuple[Union[float, int], Union[float, int]]:
    """Calculate area intersection and union of two boxes.

    Args:
        box1: Box in format (xmin, ymin, xmax, ymax).
        box2: Box in format (xmin, ymin, xmax, ymax).

    Returns:
        Intersection and Union.
    """
    # code modified from https://pytorch.org/vision/main/_modules/torchvision/ops/boxes.html
    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = np.maximum(box1[:2], box2[:2])
    rb = np.minimum(box1[2:], box2[2:])

    wh = (rb - lt).clip(min=0)
    inter = wh[0] * wh[1]

    union = area1 + area2 - inter

    return inter, union


def overlap_box_groups(
    bboxes: Union[List[NDArray], List[Tuple[float, float, float, float]]],
    iou_thr: float,
    cuda: bool
) -> Tuple[int, NDArray]:
    """Calculate groups of boxes with each box overlapping at least one in their group by at least iou_thr.

    Args:
        bboxes: List of bounding boxes in format (x1, y1, x2, y2).
        iou_thr: Intersection over union threshold.
        cuda: Calculate IoUs on GPU.

    Returns:
        Number of groups and array of group labels for each box.
    """
    bboxes = torch.tensor(np.array(bboxes))
    if cuda:
        bboxes = bboxes.to("cuda")
    graph = (t_box_iou(bboxes, bboxes) >= iou_thr).int()
    if cuda:
        graph = graph.cpu()
    n_comp, labels = connected_components(csr_array(graph.numpy()))
    return n_comp, labels
