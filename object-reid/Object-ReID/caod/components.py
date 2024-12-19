import numpy as np

from typing import Tuple
from numpy.typing import NDArray

from .tools import box_inter_union


def _adjust_to_size(x, y, w, h, resolution):
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > resolution[0]:
        x = resolution[0] - w
    if y + h > resolution[1]:
        y = resolution[1] - h
    return x, y


def combine_components(values: NDArray[np.int32], V1: int, V2: int, V3: float, resolution: Tuple[int, int]):
    """Combine BBs created using connected component search further.

    Args:
        values: Bounding Boxes in format (x, y, width, height, area).
        V1: Size goal for Boxes created by combining smaller BBs.
        V2: Minimum size for Boxes created around single small BBs.
        V3: IoU threshold used to combine two BBs, which almost form one larger box already.
        resolution: Resolution of the source image.

    Returns:
        Bounding Boxes in format (x, y, width, height, area).
    """
    values = list(values)
    values.sort(key=lambda x: x[0])
    results = []
    # greedy algorithm going through boxes from left to right to combine boxes
    while len(values) > 0:
        x, y, w, h, a = values.pop(0)
        if x < 0 or y < 0 or x+w > resolution[0] or y+h > resolution[1]:
            print(x, y, w, h)
        matches = []
        max_x = x + w
        max_y = y + h
        min_x = x
        min_y = y
        i = 0
        # don't combine large boxes further
        if w <= V1 and h <= V1:
            while i < len(values):
                x_, y_, w_, h_, a_ = values[i]
                xw = x_ + w_
                yh = y_ + h_
                # both boxes have to fit into one V1 x V1 box
                if 0 <= xw - x <= V1 and 0 <= max(max_y, yh) - min(y, y_) <= V1:
                    matches.append(values.pop(i))
                else:
                    i += 1
        if len(matches) > 0:
            for x_, y_, w_, h_, a_ in matches:
                if x_ < min_x:
                    min_x = x_
                if y_ < min_y:
                    min_y = y_
                if x_ + w_ > max_x:
                    max_x = x_ + w_
                if y_ + h_ > max_y:
                    max_y = y_ + h_
            # because boxes are not all checked against each other,
            #  final box can have size up to V1 x 2*V1
            w = max_x - min_x
            h = max_y - min_y
            assert w <= resolution[0] and h <= resolution[1], f"1, {min_x, min_y, w, h}"
            if w < V1:
                min_x = (min_x + max_x) // 2 - (V1 // 2)
                w = V1
            if h < V1:
                min_y = (min_y + max_y) // 2 - (V1 // 2)
                h = V1
            min_x, min_y = _adjust_to_size(min_x, min_y, w, h, resolution)
            results.append((min_x, min_y, w, h, w*h))
        else:
            # if single small or slim box, enlarge it
            if w < V2:
                x = x + w // 2 - (V2 // 2)
                w = V2
            if h < V2:
                y = y + h // 2 - (V2 // 2)
                h = V2
            x, y = _adjust_to_size(x, y, w, h, resolution)
            results.append((x, y, w, h, w*h))
    # combine boxes, which almost form one large box together anyways
    results_ = []
    while len(results) > 0:
        x, y, w, h, a = results.pop(0)
        i = 0
        c_flag = False
        while i < len(results):
            x_, y_, w_, h_, a_ = results[i]
            xmin = min(x, x_)
            ymin = min(y, y_)
            xmax = max(x+w, x_+w_)
            ymax = max(y+h, y_+h_)
            wmax = xmax - xmin
            hmax = ymax - ymin
            inter, union = box_inter_union(
                [x, y, x+w, y+h],
                [x_, y_, x_+w_, y_+h_]
            )
            # overlap between combined box and individual boxes
            iou = union / (wmax * hmax)
            if iou >= V3:
                results.pop(i)
                results.append((xmin, ymin, wmax, hmax, wmax * hmax))
                c_flag = True
                break
            else:
                i += 1
        if not c_flag:
            results_.append((x, y, w, h, a))
    results = results_
    results_ = []
    # filter boxes which are completely inside others
    for i, (x, y, w, h, a) in enumerate(results):
        keep = True
        for j, (x_, y_, w_, h_, a_) in enumerate(results[i:]):
            if x >= x_ and y >= y_ and x+w <= x_+w_ and y+h <= y_+h_ and i != j+i:
                keep = False
        if keep:
            results_.append((x, y, w, h, a))
    return np.array(results_, dtype=np.int32)
