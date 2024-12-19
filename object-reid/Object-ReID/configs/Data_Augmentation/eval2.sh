cd "path/to/object-reid/Object-ReID"

cfg='configs/Data_Augmentation/CO3D_v10_transfer.yml'


bsub1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" INPUT.PROB "(0.5)" INPUT.V_PROB "(0.5)" SOLVER.BASE_LR "(0.00007)" TEST.WEIGHT "('./log/Data_Augmentation/co3d_reid_v10/v5/resnet50_nl_checkpoint_32532.pt')" OUTPUT_DIR "('./log/Data_Augmentation/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'H-Flip', 'V-Flip'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" INPUT.PROB "(0.5)" INPUT.CROP "on" SOLVER.BASE_LR "(0.00007)" TEST.WEIGHT "('./log/Data_Augmentation/co3d_reid_v10/v11/resnet50_nl_checkpoint_32532.pt')" OUTPUT_DIR "('./log/Data_Augmentation/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'H-Flip', 'Crop'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" INPUT.V_PROB "(0.5)" INPUT.CROP "on" SOLVER.BASE_LR "(0.00007)" TEST.WEIGHT "('./log/Data_Augmentation/co3d_reid_v10/v10/resnet50_nl_checkpoint_32532.pt')" OUTPUT_DIR "('./log/Data_Augmentation/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'V-Flip', 'Crop'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" INPUT.PROB "(0.5)" INPUT.V_PROB "(0.5)" INPUT.CROP "on" SOLVER.BASE_LR "(0.00007)" TEST.WEIGHT "('./log/Data_Augmentation/co3d_reid_v10/v9/resnet50_nl_checkpoint_32532.pt')" OUTPUT_DIR "('./log/Data_Augmentation/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'H-Flip', 'V-Flip', 'Crop'])"
