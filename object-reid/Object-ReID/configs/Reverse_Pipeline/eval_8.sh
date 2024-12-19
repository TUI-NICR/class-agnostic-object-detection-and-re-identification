cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v48/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_reid_v1')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v48/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v4', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_co3d_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_co3d_reid_v2')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v48/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v2', 'co3d_reid_v4', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v48/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v4', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('oho_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/oho_reid_v2')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v48/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v4', 'res_128'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v54/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v10', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_reid_v1')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v54/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v10', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_co3d_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_co3d_reid_v2')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v54/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v2', 'co3d_reid_v10', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v54/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v10', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('oho_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/oho_reid_v2')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v54/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v10', 'res_128'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v11/v56/resnet50_checkpoint_33312.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v11', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_reid_v1')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v11/v56/resnet50_checkpoint_33312.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v11', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_co3d_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_co3d_reid_v2')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v11/v56/resnet50_checkpoint_33312.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v2', 'co3d_reid_v11', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v11/v56/resnet50_checkpoint_33312.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v11', 'res_128'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('oho_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/oho_reid_v2')" INPUT.IMG_SIZE "([128, 128])" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v11/v56/resnet50_checkpoint_33312.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v11', 'res_128'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 256])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v55/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v10', 'res_128', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 256])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v55/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v10', 'res_128', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_co3d_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_co3d_reid_v2')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 256])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v55/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v2', 'co3d_reid_v10', 'res_128', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 256])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v55/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v10', 'res_128', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('oho_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/oho_reid_v2')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 256])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v55/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v10', 'res_128', 'keep_ratio'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 384])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v58/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v10', 'res_192', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 384])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v58/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v10', 'res_192', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_co3d_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_co3d_reid_v2')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 384])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v58/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v2', 'co3d_reid_v10', 'res_192', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 384])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v58/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v10', 'res_192', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('oho_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/oho_reid_v2')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 384])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v58/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v10', 'res_192', 'keep_ratio'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([256, 512])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v57/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v10', 'res_256', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([256, 512])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v57/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_reid_v1', 'co3d_reid_v10', 'res_256', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_co3d_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_co3d_reid_v2')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([256, 512])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v57/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v2', 'co3d_reid_v10', 'res_256', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" OUTPUT_DIR "('./log/Reverse_Pipeline/combined_tools_redwood_reid_v1')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([256, 512])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v57/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-combined_tools_redwood_reid_v1', 'co3d_reid_v10', 'res_256', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('oho_reid_v2')" OUTPUT_DIR "('./log/Reverse_Pipeline/oho_reid_v2')" INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([256, 512])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v10/v57/resnet50_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'test-oho_reid_v2', 'co3d_reid_v10', 'res_256', 'keep_ratio'])"
sleep 5