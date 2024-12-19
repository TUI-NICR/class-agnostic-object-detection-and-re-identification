cd "path/to/object-reid/Object-ReID"

cfg='configs/Embedding_Dim/CO3D_v10_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.000035)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" TEST.WEIGHT "('./log/Embedding_Dim/co3d_reid_v12/v2/resnet50_nl_checkpoint_6650.pt')" OUTPUT_DIR "('./log/Embedding_Dim/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'dim_2048'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.00014)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" TEST.WEIGHT "('./log/Embedding_Dim/co3d_reid_v12/v0/resnet50_nl_checkpoint_3724.pt')" OUTPUT_DIR "('./log/Embedding_Dim/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'dim_2048'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.000035)" MODEL.REDUCE_DIM "('on')" MODEL.REDUCED_DIM "(1024)" MODEL.CENTER_FEAT_DIM "(1024)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" TEST.WEIGHT "('./log/Embedding_Dim/co3d_reid_v12/v1/resnet50_nl_checkpoint_4522.pt')" OUTPUT_DIR "('./log/Embedding_Dim/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'dim_1024'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.00014)" MODEL.REDUCE_DIM "('on')" MODEL.REDUCED_DIM "(1024)" MODEL.CENTER_FEAT_DIM "(1024)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" TEST.WEIGHT "('./log/Embedding_Dim/co3d_reid_v12/v3/resnet50_nl_checkpoint_3458.pt')" OUTPUT_DIR "('./log/Embedding_Dim/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'dim_1024'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.000035)" MODEL.REDUCE_DIM "('on')" MODEL.REDUCED_DIM "(256)" MODEL.CENTER_FEAT_DIM "(256)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" TEST.WEIGHT "('./log/Embedding_Dim/co3d_reid_v12/v4/resnet50_nl_checkpoint_3458.pt')" OUTPUT_DIR "('./log/Embedding_Dim/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'dim_256'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg DATASETS.NAMES "('combined_tools_redwood_reid_v1')" SOLVER.BASE_LR "(0.00014)" MODEL.REDUCE_DIM "('on')" MODEL.REDUCED_DIM "(256)" MODEL.CENTER_FEAT_DIM "(256)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" TEST.WEIGHT "('./log/Embedding_Dim/co3d_reid_v12/v5/resnet50_nl_checkpoint_3458.pt')" OUTPUT_DIR "('./log/Embedding_Dim/combined_tools_redwood_reid_v1')" WANDB.TAGS "(['test', 'combined_tools_redwood_reid_v1', 'dim_256'])"
