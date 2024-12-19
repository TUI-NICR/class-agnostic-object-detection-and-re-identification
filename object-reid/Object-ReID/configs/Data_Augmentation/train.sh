cd "path/to/object-reid/Object-ReID"

cfg='configs/Data_Augmentation/CO3D_v10_baseline.yml'


batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" WANDB.TAGS "(['train', 'co3d_reid_v10'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.PROB "(0.5)" SOLVER.BASE_LR "(0.00007)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'H-Flip'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.V_PROB "(0.5)" SOLVER.BASE_LR "(0.00007)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'V-Flip'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.RE_PROB "(0.5)" SOLVER.BASE_LR "(0.00007)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'RE'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.CROP "on" SOLVER.BASE_LR "(0.00007)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'Crop'])"
