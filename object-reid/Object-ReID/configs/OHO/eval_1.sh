cd "path/to/object-reid/Object-ReID"

cfg='configs/OHO/plain_resnet_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v67/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v1', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v68/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v1', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v69/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v1', 'res_256'])"
