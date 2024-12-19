cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_boxcrop_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([128, 128])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v9/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_128', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([192, 192])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v10/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_192', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
