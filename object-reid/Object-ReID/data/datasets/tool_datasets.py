from .combined_dataset import CombinedImageDataset


class KTH_ReID_v1(CombinedImageDataset):
    """
    ReiD tool dataset created from KTH dataset.
    Evaluation Only!

    # identities:     9
    # images train:   0
    # images test:  108
    # images query:  27
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(KTH_ReID_v1, self).__init__(
            train_splits=[],
            query_splits=["kth_reid_v1"],
            name="kth_reid_v1", root=root, verbose=verbose
        )


class WorkingHands_ReID_v1(CombinedImageDataset):
    """
    ReiD tool dataset created from WorkingHands dataset.
    Evaluation Only!

    # identities:    11
    # images train:   0
    # images test:  132
    # images query:  33
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(WorkingHands_ReID_v1, self).__init__(
            train_splits=[],
            query_splits=["workinghands_reid_v1"],
            name="workinghands_reid_v1", root=root, verbose=verbose
        )


class WorkingHands_ReID_v2(CombinedImageDataset):
    """
    Improved (but smaller) version of WorkingHands_ReID_v1.
    Evaluation Only!

    # identities:     5
    # images train:   0
    # images test:   60
    # images query:  15
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(WorkingHands_ReID_v2, self).__init__(
            train_splits=[],
            query_splits=["workinghands_reid_v2"],
            name="workinghands_reid_v2", root=root, verbose=verbose
        )


class Attach_ReID_v1(CombinedImageDataset):
    """
    ReiD tool dataset created from E4SM Attach dataset.
    Evaluation Only!

    # identities:     3
    # images train:   0
    # images test:   36
    # images query:   9
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(Attach_ReID_v1, self).__init__(
            train_splits=[],
            query_splits=["attach_reid_v1"],
            name="attach_reid_v1", root=root, verbose=verbose
        )


class Attach_ReID_v2(CombinedImageDataset):
    """
    Improved version of Attach_ReID_v1.
    Evaluation Only!

    # identities:     3
    # images train:   0
    # images test:   36
    # images query:   9
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(Attach_ReID_v2, self).__init__(
            train_splits=[],
            query_splits=["attach_reid_v2"],
            name="attach_reid_v2", root=root, verbose=verbose
        )


class CombinedTools_v1(CombinedImageDataset):
    """
    Combined ReID tool dataset created from KTH, WorkingHands and E4SM Attach datasets.
    Evaluation Only!

    # identities:    23
    # images train:   0
    # images test:  276
    # images query:  69
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(CombinedTools_v1, self).__init__(
            train_splits=[],
            query_splits=["kth_reid_v1", "workinghands_reid_v1", "attach_reid_v1"],
            name="combined_tools_reid_v1", root=root, verbose=verbose
        )


class CombinedTools_v2(CombinedImageDataset):
    """
    Improved (but smaller) ReID tool dataset created from KTH, WorkingHands and E4SM Attach datasets.
    Evaluation Only!

    # identities:    17
    # images train:   0
    # images test:  204
    # images query:  51
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(CombinedTools_v2, self).__init__(
            train_splits=[],
            query_splits=["kth_reid_v1", "workinghands_reid_v2", "attach_reid_v2"],
            name="combined_tools_reid_v2", root=root, verbose=verbose
        )


class CombinedToolsCO3D_v1(CombinedImageDataset):
    """
    Combined ReID tool dataset created from KTH, WorkingHands and E4SM Attach datasets.
    Includes all CO3D gallery samples to increase evaluation difficulty.
    Also includes CO3D train split for convenience.
    Evaluation Only!

    # identities:    23
    # images train:   0 + 161255
    # images test:  276 + 134681
    # images query:  68
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(CombinedToolsCO3D_v1, self).__init__(
            train_splits=["co3d_reid_v1"],
            query_splits=["kth_reid_v1", "workinghands_reid_v1", "attach_reid_v1"],
            add_gallery_splits=["co3d_reid_v1"],
            name="combined_tools_co3d_reid_v1", root=root, verbose=verbose
        )


class CombinedToolsCO3D_v2(CombinedImageDataset):
    """
    Improved ReID tool dataset created from KTH, WorkingHands and E4SM Attach datasets.
    Includes all CO3D gallery samples to increase evaluation difficulty.
    Also includes CO3D train split for convenience.
    Evaluation Only!

    # identities:    17
    # images train:   0 + 161255
    # images test:  204 + 134681
    # images query:  51
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(CombinedToolsCO3D_v2, self).__init__(
            train_splits=["co3d_reid_v1"],
            query_splits=["kth_reid_v1", "workinghands_reid_v2", "attach_reid_v2"],
            add_gallery_splits=["co3d_reid_v1"],
            name="combined_tools_co3d_reid_v2", root=root, verbose=verbose
        )


class CombinedToolsRedwood_v1(CombinedImageDataset):
    """
    Improved ReID tool dataset created from KTH, WorkingHands and E4SM Attach datasets.
    Includes all Redwood gallery samples to increase evaluation difficulty.
    Also includes Redwood train split for convenience.
    Evaluation Only!

    # identities:    17
    # images train:   0 + 111045
    # images test:  204 +  85746
    # images query:  51
    """
    def __init__(self, root='./toDataset', verbose=True):
        super(CombinedToolsRedwood_v1, self).__init__(
            train_splits=["redwood_reid_v1"],
            query_splits=["kth_reid_v1", "workinghands_reid_v2", "attach_reid_v2"],
            add_gallery_splits=["redwood_reid_v1"],
            name="combined_tools_redwood_reid_v1", root=root, verbose=verbose
        )
