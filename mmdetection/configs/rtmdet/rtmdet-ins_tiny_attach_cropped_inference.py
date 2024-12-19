_base_ = './rtmdet-ins_tiny_attach.py'

input_size_tuple = (768, 576)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Crop',
         crop_x1=351,
         crop_y1=0,
         crop_x2=2159,
         crop_y2=1440),
    dict(type='Resize', scale=(input_size_tuple[0], input_size_tuple[0]), keep_ratio=True),
    dict(type='Pad', size=(input_size_tuple[0], input_size_tuple[0]), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader
