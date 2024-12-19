cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet.yml'


batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('co3d_reid_v1')" INPUT.IMG_SIZE "([64, 64])" OUTPUT_DIR "('./log/Reverse_Pipeline/co3d_reid_v1')" WANDB.TAGS "(['train', 'co3d_reid_v1', 'res_64'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('co3d_reid_v1')" INPUT.IMG_SIZE "([32, 32])" OUTPUT_DIR "('./log/Reverse_Pipeline/co3d_reid_v1')" WANDB.TAGS "(['train', 'co3d_reid_v1', 'res_32'])"
