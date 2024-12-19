cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet.yml'


batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('co3d_reid_v1')" INPUT.IMG_SIZE "([128, 128])" WANDB.TAGS "(['train', 'co3d_reid_v1', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('co3d_reid_v1')" INPUT.IMG_SIZE "([192, 192])" WANDB.TAGS "(['train', 'co3d_reid_v1', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('co3d_reid_v1')" INPUT.IMG_SIZE "([256, 256])" WANDB.TAGS "(['train', 'co3d_reid_v1', 'res_256'])"
