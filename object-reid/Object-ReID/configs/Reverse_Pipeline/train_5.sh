cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet.yml'


batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('co3d_reid_v10')" INPUT.IMG_SIZE "([192, 192])" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('co3d_reid_v10')" INPUT.IMG_SIZE "([256, 256])" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_256'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('co3d_reid_v11')" INPUT.IMG_SIZE "([192, 192])" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('co3d_reid_v11')" INPUT.IMG_SIZE "([256, 256])" WANDB.TAGS "(['train', 'co3d_reid_v4', 'res_256'])"