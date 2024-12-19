cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v67/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v1', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v68/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v1', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([256, 256])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v1/v69/resnet50_checkpoint_29172.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v1', 'res_256'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v52/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v10', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([256, 256])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v53/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v10', 'res_256'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v11/v54/resnet50_checkpoint_33312.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v11', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([256, 256])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v11/v55/resnet50_checkpoint_33312.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v11', 'res_256'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 256])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v47/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v4', 'res_128', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 384])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v45/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v4', 'res_192', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([256, 512])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v46/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v4', 'res_256', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 512])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v47/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v4', 'res_128', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 768])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v45/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v4', 'res_192', 'keep_ratio'])"
