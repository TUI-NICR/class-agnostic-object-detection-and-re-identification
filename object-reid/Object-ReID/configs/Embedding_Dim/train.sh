cd "path/to/object-reid/Object-ReID"

cfg='configs/Embedding_Dim/CO3D_v10_baseline.yml'


batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" DATASETS.NAMES "('co3d_reid_v12')" OUTPUT_DIR "('./log/Embedding_Dim/co3d_reid_v12')" WANDB.TAGS "(['train', 'co3d_reid_v12', 'dim_2048'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" DATASETS.NAMES "('co3d_reid_v12')" OUTPUT_DIR "('./log/Embedding_Dim/co3d_reid_v12')" WANDB.TAGS "(['train', 'co3d_reid_v12', 'dim_2048'])"
sleep 5


batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" MODEL.REDUCE_DIM "('on')" MODEL.REDUCED_DIM "(1024)" MODEL.CENTER_FEAT_DIM "(1024)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" DATASETS.NAMES "('co3d_reid_v12')" OUTPUT_DIR "('./log/Embedding_Dim/co3d_reid_v12')" WANDB.TAGS "(['train', 'co3d_reid_v12', 'dim_1024'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" MODEL.REDUCE_DIM "('on')" MODEL.REDUCED_DIM "(1024)" MODEL.CENTER_FEAT_DIM "(1024)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" DATASETS.NAMES "('co3d_reid_v12')" OUTPUT_DIR "('./log/Embedding_Dim/co3d_reid_v12')" WANDB.TAGS "(['train', 'co3d_reid_v12', 'dim_1024'])"
sleep 5


batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.000035)" MODEL.REDUCE_DIM "('on')" MODEL.REDUCED_DIM "(256)" MODEL.CENTER_FEAT_DIM "(256)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" DATASETS.NAMES "('co3d_reid_v12')" OUTPUT_DIR "('./log/Embedding_Dim/co3d_reid_v12')" WANDB.TAGS "(['train', 'co3d_reid_v12', 'dim_256'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg SOLVER.BASE_LR "(0.00014)" MODEL.REDUCE_DIM "('on')" MODEL.REDUCED_DIM "(256)" MODEL.CENTER_FEAT_DIM "(256)" SOLVER.MAX_EPOCHS "(120)" SOLVER.STEPS "([70, 100])" SOLVER.WARMUP_FACTOR "(0.01)" SOLVER.WARMUP_ITERS "(10)" DATASETS.NAMES "('co3d_reid_v12')" OUTPUT_DIR "('./log/Embedding_Dim/co3d_reid_v12')" WANDB.TAGS "(['train', 'co3d_reid_v12', 'dim_256'])"
