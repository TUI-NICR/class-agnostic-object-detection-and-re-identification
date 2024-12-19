cd "path/to/object-reid/Object-ReID"

cfg='configs/Sampling/CO3D_v10_transfer_log.yml'


batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(4)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v0/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_4'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(4)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v1/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_4'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" DATALOADER.NUM_CLASS "(4)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v8/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_4'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(1)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v6/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_1'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(1)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v7/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_1'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" DATALOADER.NUM_CLASS "(1)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v9/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_1'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(16)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v10/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_16'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(16)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v2/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_16'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" DATALOADER.NUM_CLASS "(16)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v3/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_16'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(8)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v4/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_8'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(8)" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v5/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['train', 'co3d_reid_v10', 'P_4', 'B_64', 'C_8'])"
