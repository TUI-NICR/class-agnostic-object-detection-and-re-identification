#!/usr/bin/env python
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH -t 7:30:00
#SBATCH --exclude=jupiter2,titan,titan2,saturn8i
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=henning.franke@tu-ilmenau.de

# encoding: utf-8

"""
Main entry point script for ReID-Survey Repository. Used for training, evaluation and inference.
"""

import argparse
import os
import sys
import torch
import re

from torch.backends import cudnn
from time import sleep
from numpy.random import default_rng

sys.path = [p for p in sys.path if not "mmdetection" in p]
sys.path.append(os.getcwd())
sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from survey.utils.lr_scheduler import WarmupMultiStepLR
from survey.utils.logger import setup_logger
from tools.train import do_train
from tools.test import do_test
from tools.inference import do_inference


def main():
    parser = argparse.ArgumentParser(description="AGW Re-ID Baseline")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # This part is needed because with cfg.OUTPUT_VERSIONING a new folder is created in output_dir.
    # The name of that folder depends on the already existing folders. If multiple Slurm runs start at the
    # same time they will get in each others way because output_dir isn't locked by the file system between
    # reading the existing folders and creating a new one. The delay is large enough that errors should be minimal.
    # If one occurs the run will crash with OSError from os.mkdir.
    rng = default_rng()
    delay = rng.random()*5
    sleep(delay)

    # Create output folder
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if cfg.OUTPUT_VERSIONING == 'on':
        assert output_dir
        while True:
            current_version = -1
            for f in os.listdir(output_dir):
                if os.path.isdir(os.path.join(output_dir, f)) and re.search("v[0-9]+", f).span() == (0, len(f)):
                    version = int(f[1:])
                    if version > current_version:
                        current_version = version
            output_dir_ = os.path.join(output_dir, f"v{current_version+1}")
            if not os.path.isdir(output_dir_):
                break
        os.mkdir(output_dir_)
        output_dir = output_dir_

    # setup logger
    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    # read config
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # I'm not sure this code does anything
    #if cfg.MODEL.DEVICE == "cuda":
    #    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True

    # create dataloader and model from config
    data_loader, num_query, num_classes, num_inference_samples, num_superclasses = make_data_loader(cfg)
    model = build_model(cfg, num_classes, num_superclasses=num_superclasses)

    # model to GPU
    if 'cpu' not in cfg.MODEL.DEVICE:
        #if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model)
        model.to(device=cfg.MODEL.DEVICE)

    # do ONLY inference and exit
    if cfg.INFERENCE.DO_INFERENCE == 'on':
        logger.info("Inference Only")
        model.load_param(cfg.TEST.WEIGHT)
        do_inference(cfg, model, data_loader, num_inference_samples)
        return

    # do ONLY evaluation and exit
    if cfg.TEST.EVALUATE_ONLY == 'on':
        logger.info("Evaluate Only")
        model.load_param(cfg.TEST.WEIGHT)
        do_test(cfg, output_dir, model, data_loader, num_query)
        return

    # create loss functions and optimizer
    criterion = model.get_creterion(cfg, num_classes, num_superclasses=num_superclasses)
    optimizer = model.get_optimizer(cfg, criterion)

    # This Code governs behaviour when resuming training from checkpoint
    # I'm not sure if it even works as I have never used it.
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
        print('Path to the checkpoint of center_param:', path_to_center_param)
        path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
        print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
        optimizer['model'].load_state_dict(torch.load(path_to_optimizer))
        criterion['center'].load_state_dict(torch.load(path_to_center_param))
        optimizer['center'].load_state_dict(torch.load(path_to_optimizer_center))
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    # create LR Scheduler and load pretrained ReID weights (not just CNN backbone)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'other':
        start_epoch = 0
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH)["model"])
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    # create LR Scheduler, loading CNN backbone weights already happened in build_model
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    else:
        print('Only support pretrain_choice for imagenet, self and other, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    # do training and validation
    do_train(
        cfg,
        output_dir,
        model,
        data_loader,
        optimizer,
        scheduler,
        criterion,
        num_query,
        start_epoch
    )


if __name__ == '__main__':
    main()
