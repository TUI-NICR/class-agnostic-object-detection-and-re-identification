cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_boxcrop_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_2372.pt')" WANDB.TAGS "(['test', 'CP_1', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_4744.pt')" WANDB.TAGS "(['test', 'CP_2', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_7116.pt')" WANDB.TAGS "(['test', 'CP_3', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_9488.pt')" WANDB.TAGS "(['test', 'CP_4', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_11860.pt')" WANDB.TAGS "(['test', 'CP_5', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_14232.pt')" WANDB.TAGS "(['test', 'CP_6', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_16604.pt')" WANDB.TAGS "(['test', 'CP_7', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_18976.pt')" WANDB.TAGS "(['test', 'CP_8', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_21348.pt')" WANDB.TAGS "(['test', 'CP_9', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_23720.pt')" WANDB.TAGS "(['test', 'CP_10', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" DATASETS.NAMES "('combined_tools_reid_v1')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v11/resnet50_checkpoint_26092.pt')" WANDB.TAGS "(['test', 'CP_11', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_256', 'box_crop'])"
