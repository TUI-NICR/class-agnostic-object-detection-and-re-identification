cd "path/to/object-reid/Object-ReID"

cfg='configs/Num_Classes/CO3D_v10_transfer.yml'


bsub1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.00007)" SOLVER.MAX_EPOCHS "(24)" SOLVER.STEPS "([14, 20])" TEST.WEIGHT "('./log/Num_Classes/co3d_reid_v13/v0/resnet50_nl_checkpoint_37080.pt')" OUTPUT_DIR "('./log/Num_Classes/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'co3d_reid_v13', 'combined_tools_redwood_reid_v1'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.00007)" SOLVER.MAX_EPOCHS "(24)" SOLVER.STEPS "([14, 20])" TEST.WEIGHT "('./log/Num_Classes/co3d_reid_v14/v0/resnet50_nl_checkpoint_37104.pt')" OUTPUT_DIR "('./log/Num_Classes/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'co3d_reid_v14', 'combined_tools_redwood_reid_v1'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.00007)" SOLVER.MAX_EPOCHS "(24)" SOLVER.STEPS "([14, 20])" TEST.WEIGHT "('./log/Num_Classes/co3d_reid_v15/v0/resnet50_nl_checkpoint_37248.pt')" OUTPUT_DIR "('./log/Num_Classes/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'co3d_reid_v15', 'combined_tools_redwood_reid_v1'])"
