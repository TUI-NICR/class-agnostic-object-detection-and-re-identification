cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v52/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v10', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v53/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v10', 'res_256'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v54/resnet50_checkpoint_33312.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v11', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v55/resnet50_checkpoint_33312.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v11', 'res_256'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v52/resnet50_checkpoint_32532.pt')" DATASETS.NAMES "('combined_tools_reid_v1')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v10', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v53/resnet50_checkpoint_32532.pt')" DATASETS.NAMES "('combined_tools_reid_v1')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v10', 'res_256'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v54/resnet50_checkpoint_33312.pt')" DATASETS.NAMES "('combined_tools_reid_v1')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v11', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v55/resnet50_checkpoint_33312.pt')" DATASETS.NAMES "('combined_tools_reid_v1')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v11', 'res_256'])"
sleep 5