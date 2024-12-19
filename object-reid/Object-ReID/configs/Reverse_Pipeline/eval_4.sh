cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 256])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v47/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_128', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 384])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v45/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_192', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([256, 512])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v46/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_256', 'keep_ratio'])"
sleep 5

batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([128, 512])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v47/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_128', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([192, 768])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v45/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_192', 'keep_ratio'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.KEEP_RATIO "('on')" INPUT.IMG_SIZE "([256, 1024])" INPUT.CROP "('off')" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v46/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_256', 'keep_ratio'])"
