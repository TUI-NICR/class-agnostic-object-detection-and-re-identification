cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 256])" INPUT.CROP "('off')" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_128', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 384])" INPUT.CROP "('off')" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_192', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([256, 512])" INPUT.CROP "('off')" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_256', 'keep_ratio'])"
