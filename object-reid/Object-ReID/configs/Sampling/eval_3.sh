cd "path/to/object-reid/Object-ReID"

cfg='configs/Sampling/CO3D_v10_transfer_redwood.yml'


batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(16)" INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v30/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'co3d_reid_v10', 'test-combined_tools_redwood_reid_v1', 'P_4', 'B_64', 'C_16', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(16)" INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v27/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'co3d_reid_v10', 'test-combined_tools_redwood_reid_v1', 'P_4', 'B_64', 'C_16', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" DATALOADER.NUM_CLASS "(16)" INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v29/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'co3d_reid_v10', 'test-combined_tools_redwood_reid_v1', 'P_4', 'B_64', 'C_16', 'res_192'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00007)" DATALOADER.NUM_CLASS "(8)" INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v25/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'co3d_reid_v10', 'test-combined_tools_redwood_reid_v1', 'P_4', 'B_64', 'C_8', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" DATALOADER.NUM_CLASS "(8)" INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v28/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'co3d_reid_v10', 'test-combined_tools_redwood_reid_v1', 'P_4', 'B_64', 'C_8', 'res_192'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" DATALOADER.NUM_CLASS "(8)" INPUT.IMG_SIZE "([192, 192])" TEST.WEIGHT "('./log/Sampling/co3d_reid_v10/v26/resnet50_nl_checkpoint_32532.pt')" WANDB.TAGS "(['test', 'co3d_reid_v10', 'test-combined_tools_redwood_reid_v1', 'P_4', 'B_64', 'C_8', 'res_192'])"
