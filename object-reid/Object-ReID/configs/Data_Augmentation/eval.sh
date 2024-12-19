cd "path/to/object-reid/Object-ReID"

cfg='configs/Data_Augmentation/CO3D_v10_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.00007)" TEST.WEIGHT "('./log/Data_Augmentation/co3d_reid_v10/v0/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" INPUT.PROB "(0.5)" SOLVER.BASE_LR "(0.00007)" TEST.WEIGHT "('./log/Data_Augmentation/co3d_reid_v10/v1/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'H-Flip'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" INPUT.V_PROB "(0.5)" SOLVER.BASE_LR "(0.00007)" TEST.WEIGHT "('./log/Data_Augmentation/co3d_reid_v10/v2/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'V-Flip'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" INPUT.RE_PROB "(0.5)" SOLVER.BASE_LR "(0.00007)" TEST.WEIGHT "('./log/Data_Augmentation/co3d_reid_v10/v3/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'RE'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" INPUT.CROP "on" SOLVER.BASE_LR "(0.00007)" TEST.WEIGHT "('./log/Data_Augmentation/co3d_reid_v10/v4/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'Crop'])"
