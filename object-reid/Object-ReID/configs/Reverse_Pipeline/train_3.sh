cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" INPUT.PADDING "(20)" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_256', 'pad_20'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" INPUT.PADDING "(40)" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_256', 'pad_40'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" INPUT.PADDING "(80)" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_256', 'pad_80'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" INPUT.PADDING "(160)" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_256', 'pad_160'])"
