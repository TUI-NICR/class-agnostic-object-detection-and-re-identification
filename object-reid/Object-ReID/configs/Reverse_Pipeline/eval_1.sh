cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v3/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'co3d_reid_v4', 'test-combined_tools_co3d_reid_v1', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v4/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'co3d_reid_v4', 'test-combined_tools_co3d_reid_v1', 'res_256'])"
