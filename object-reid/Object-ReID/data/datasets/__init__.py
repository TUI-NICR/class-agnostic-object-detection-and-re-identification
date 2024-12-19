# encoding: utf-8
from .dataset_loader import ImageDataset
from .object_reid_datasets import ObjectReIDDataset
from .object_reid_datasets import CO3D_ReID_v1, CO3D_ReID_v2, GoogleScan_ReID_v1, Redwood_ReID_v1, Redwood_ReID_COCO, CO3D_ReID_Masked_v1, Combined_Redwood_CO3D_v1, OHO_ReID_v1, CO3D_ReID_v3, OHO_ReID_v2, CO3D_ReID_v4, CO3D_ReID_v5, CO3D_ReID_v6, CO3D_ReID_v7, CO3D_ReID_v8, CO3D_ReID_v10, CO3D_ReID_v11, CO3D_ReID_v12, CO3D_ReID_v13, CO3D_ReID_v14, CO3D_ReID_v15
from .tool_datasets import KTH_ReID_v1, WorkingHands_ReID_v1, Attach_ReID_v1, CombinedTools_v1, CombinedToolsCO3D_v1, WorkingHands_ReID_v2, Attach_ReID_v2, CombinedTools_v2, CombinedToolsCO3D_v2, CombinedToolsRedwood_v1

from survey.data.datasets.dukemtmcreid import DukeMTMCreID
from survey.data.datasets.market1501 import Market1501
from survey.data.datasets.msmt17 import MSMT17
from survey.data.datasets.veri import VeRi
from survey.data.datasets.partial_ilids import PartialILIDS
from survey.data.datasets.partial_reid import PartialREID


__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
    'partial_reid': PartialREID,
    'partial_ilids': PartialILIDS,
    'co3d_reid_v1': CO3D_ReID_v1,
    'co3d_reid_masked_v1': CO3D_ReID_Masked_v1,
    'co3d_reid_v2': CO3D_ReID_v2,
    'co3d_reid_v3': CO3D_ReID_v3,
    'co3d_reid_v4': CO3D_ReID_v4,
    'co3d_reid_v5': CO3D_ReID_v5,
    'co3d_reid_v6': CO3D_ReID_v6,
    'co3d_reid_v7': CO3D_ReID_v7,
    'co3d_reid_v8': CO3D_ReID_v8,
    'co3d_reid_v10': CO3D_ReID_v10,
    'co3d_reid_v11': CO3D_ReID_v11,
    'co3d_reid_v12': CO3D_ReID_v12,
    'co3d_reid_v13': CO3D_ReID_v13,
    'co3d_reid_v14': CO3D_ReID_v14,
    'co3d_reid_v15': CO3D_ReID_v15,
    'google-scan_reid_v1': GoogleScan_ReID_v1,
    'redwood_reid_v1': Redwood_ReID_v1,
    'redwood_reid_coco': Redwood_ReID_COCO,
    'combined_redwood_co3d_reid_v1': Combined_Redwood_CO3D_v1,
    'kth_reid_v1': KTH_ReID_v1,
    'workinghands_reid_v1': WorkingHands_ReID_v1,
    'workinghands_reid_v2': WorkingHands_ReID_v2,
    'attach_reid_v1': Attach_ReID_v1,
    'attach_reid_v2': Attach_ReID_v2,
    'combined_tools_reid_v1': CombinedTools_v1,
    'combined_tools_reid_v2': CombinedTools_v2,
    'combined_tools_co3d_reid_v1': CombinedToolsCO3D_v1,
    'combined_tools_co3d_reid_v2': CombinedToolsCO3D_v2,
    'combined_tools_redwood_reid_v1': CombinedToolsRedwood_v1,
    'oho_reid_v1': OHO_ReID_v1,
    'oho_reid_v2': OHO_ReID_v2
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
