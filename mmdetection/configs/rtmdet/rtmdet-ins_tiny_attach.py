_base_ = './rtmdet-ins_tiny_8xb32-300e_coco.py'

work_dir = '/path/to/attach_detection/attach_rtmdet'

dataset_type = 'CocoDataset'
data_root = '/path/to/Attach/'
classes=(
    'Screwdriver',
    'Leg',
    'WallSpacerTop',
    'ScrewNoHead',
    'Wrench',
    'Board',
    'ScrewWithHead',
    'ThreadedTupeFemale',
    'ThreadedRod',
    'Manual',
    'Hammer',
    'WallSpacerMounting'
)
model = dict(
    bbox_head=dict(
        num_classes=12,
    )
)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'

input_size_tuple = (768, 576)
# input_size_tuple = (896, 672)
# input_size_tuple = (1024, 768)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=input_size_tuple,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=input_size_tuple,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=input_size_tuple, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
train_pipeline_stage2 = train_pipeline

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(input_size_tuple[0], input_size_tuple[0]), keep_ratio=True),
    dict(type='Pad', size=(input_size_tuple[0], input_size_tuple[0]), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

batch_size = 8

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='labels/labels_train_2023_05_11.json',
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline,
        )
    )
val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='labels/labels_test_2023_05_11.json',
        data_prefix=dict(img=data_root),
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

max_epochs = 12
stage2_num_epochs = 2
base_lr = 0.002
interval = 2

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

val_evaluator = dict(
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'labels/labels_test_2023_05_11.json',
)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = dict(
    _delete_=True,
    type='OneCycleLR',
    eta_max=base_lr,
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
    convert_to_iter_based=True
)

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ),
)
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2),
]

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs={'entity': 'REDACTED', 'project': 'REDACTED'},
    ),
]