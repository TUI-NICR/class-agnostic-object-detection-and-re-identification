cd "path/to/object-reid/Object-ReID"

cfg='configs/CBIR/cbir_cfg.yml'

python tools/test_cbir.py --config_file=$cfg TEST.DATASET "('combined_tools_reid_v2')"

python tools/test_cbir.py --config_file=$cfg TEST.DATASET "('combined_tools_redwood_reid_v1')"

python tools/test_cbir.py --config_file=$cfg TEST.DATASET "('combined_tools_co3d_reid_v2')"
