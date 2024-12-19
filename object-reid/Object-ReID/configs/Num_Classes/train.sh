cd "path/to/object-reid/Object-ReID"

cfg='configs/Num_Classes/CO3D_v10_baseline.yml'


bsub1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATASETS.NAMES "('co3d_reid_v13')" OUTPUT_DIR "('./log/Num_Classes/co3d_reid_v13')" SOLVER.MAX_EPOCHS "(24)" SOLVER.STEPS "([14, 20])" WANDB.TAGS "(['train', 'co3d_reid_v13'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATASETS.NAMES "('co3d_reid_v14')" OUTPUT_DIR "('./log/Num_Classes/co3d_reid_v14')" SOLVER.MAX_EPOCHS "(24)" SOLVER.STEPS "([14, 20])" WANDB.TAGS "(['train', 'co3d_reid_v14'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATASETS.NAMES "('co3d_reid_v15')" OUTPUT_DIR "('./log/Num_Classes/co3d_reid_v15')" SOLVER.MAX_EPOCHS "(24)" SOLVER.STEPS "([14, 20])" WANDB.TAGS "(['train', 'co3d_reid_v15'])"
