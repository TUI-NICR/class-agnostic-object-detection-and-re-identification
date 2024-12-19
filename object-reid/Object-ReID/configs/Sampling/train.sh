cd "path/to/object-reid/Object-ReID"

cfg='configs/Sampling/CO3D_v10_baseline.yml'


batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(4)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_4'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(4)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_4'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" DATALOADER.NUM_CLASS "(4)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_4'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(1)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_1'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(1)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_1'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" DATALOADER.NUM_CLASS "(1)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_1'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(16)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_16'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(16)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_16'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" DATALOADER.NUM_CLASS "(16)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_16'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(8)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_8'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(8)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_8'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" DATALOADER.NUM_CLASS "(8)" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_8'])"
