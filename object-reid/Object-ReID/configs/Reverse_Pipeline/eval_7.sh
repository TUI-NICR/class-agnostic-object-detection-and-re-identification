cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([64, 64])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v67/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v1', 'res_64'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([32, 32])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v68/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v1', 'res_32'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_reid_v1')" INPUT.IMG_SIZE "([64, 64])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v67/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v1', 'res_64'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_reid_v1')" INPUT.IMG_SIZE "([32, 32])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v68/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v1', 'res_32'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_co3d_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_co3d_reid_v2')" INPUT.IMG_SIZE "([64, 64])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v67/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v2', 'co3d_reid_v1', 'res_64'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_co3d_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_co3d_reid_v2')" INPUT.IMG_SIZE "([32, 32])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v68/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v2', 'co3d_reid_v1', 'res_32'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([64, 64])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v67/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v1', 'res_64'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([32, 32])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v68/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v1', 'res_32'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('oho_reid_v2')" OUTPUT_DIR "('./log/OHO/oho_reid_v2')" INPUT.IMG_SIZE "([64, 64])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v67/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v1', 'res_64'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('oho_reid_v2')" OUTPUT_DIR "('./log/OHO/oho_reid_v2')" INPUT.IMG_SIZE "([32, 32])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v68/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v1', 'res_32'])"
