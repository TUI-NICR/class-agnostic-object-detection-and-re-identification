cd "path/to/object-reid/Object-ReID"

cfg='configs/Data_Augmentation/CO3D_v10_baseline.yml'


bsub1gpu python tools/main.py --config_file=$cfg INPUT.PROB "(0.5)" INPUT.V_PROB "(0.5)" SOLVER.BASE_LR "(0.00007)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'H-Flip', 'V-Flip'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg INPUT.PROB "(0.5)" INPUT.CROP "on" SOLVER.BASE_LR "(0.00007)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'H-Flip', 'Crop'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg INPUT.V_PROB "(0.5)" INPUT.CROP "on" SOLVER.BASE_LR "(0.00007)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'V-Flip', 'Crop'])"
sleep 5
bsub1gpu python tools/main.py --config_file=$cfg INPUT.PROB "(0.5)" INPUT.V_PROB "(0.5)" INPUT.CROP "on" SOLVER.BASE_LR "(0.00007)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'H-Flip', 'V-Flip', 'Crop'])"
