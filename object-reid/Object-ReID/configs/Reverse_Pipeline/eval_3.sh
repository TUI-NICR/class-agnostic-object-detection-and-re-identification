cd "path/to/object-reid/Object-ReID"

cfg='configs/Reverse_Pipeline/plain_resnet_transfer.yml'


batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" INPUT.PADDING "(20)" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v23/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_256', 'pad_20'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" INPUT.PADDING "(40)" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v24/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_256', 'pad_40'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" INPUT.PADDING "(80)" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v25/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_256', 'pad_80'])"
sleep 5
batch.1gpu python tools/main.py --config_file=$cfg INPUT.IMG_SIZE "([256, 256])" INPUT.PADDING "(160)" TEST.WEIGHT "('./log/Reverse_Pipeline/co3d_reid_v4/v26/resnet50_checkpoint_28464.pt')" WANDB.TAGS "(['test', 'test-combined_tools_co3d_reid_v1', 'co3d_reid_v4', 'res_256', 'pad_160'])"
