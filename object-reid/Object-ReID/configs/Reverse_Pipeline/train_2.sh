cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_boxcrop.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([128, 128])" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_128', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([192, 192])" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_192', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_256', 'box_crop'])"
