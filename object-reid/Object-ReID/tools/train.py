# encoding: utf-8

import logging
from nicr_cluster_utils.callbacks import WebLogger  # proprietary, remove or replace with another Logger

import wandb
import torch
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.logger_wandb import setup_wandb, log_eval_wandb
from utils.reid_metric import r1_mAP_mINP
from .test import create_supervised_evaluator

global ITER
ITER = 0


def create_supervised_trainer(model, optimizer, criterion, cetner_loss_weight=0.0, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (dict - class:`torch.optim.Optimizer`): the optimizer to use
        criterion (dict - class:loss function): the loss function to use
        cetner_loss_weight (float, optional): the weight for cetner_loss_weight
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """

    def _update(engine, batch):
        model.train()
        optimizer['model'].zero_grad()

        if 'center' in optimizer.keys():
            optimizer['center'].zero_grad()

        img, target, other = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        if isinstance(other[0], torch.Tensor):
            target_sc = other[0].to(device) if torch.cuda.device_count() >= 1 else other[0]
        else:
            target_sc = None
        score, feat, bn_feat, weight = model(img)
        loss = criterion['total'](score, feat, target, target_sc=target_sc, bn_feat=bn_feat, weight=weight)
        loss.backward()
        optimizer['model'].step()

        if 'center' in optimizer.keys():
            for param in criterion['center'].parameters():
                param.grad.data *= (1. / cetner_loss_weight)
            optimizer['center'].step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def do_train(
        cfg,
        output_dir,
        model,
        data_loader,
        optimizer,
        scheduler,
        criterion,
        num_query,
        start_epoch
):
    """
    Do training and validation on dataset, log results and create checkpoint weights.

    Args:
    - cfg (CfgNode): Config of experiment
    - output_dir (str): Path to output directory
    - model (Module): Loaded torch model
    - data_loader (dict): Dict containing DataLoaders at keys "train" and "eval"
    - optimizer (dict): Dict of Optimizers at keys "model" and "center" used to optimize model and center loss cluster parameters
    - scheduler (LRScheduler): Learning rate scheduler
    - criterion (dict): Dict of losses (Modules). Possible keys: "xent" for classification of IDs, "triplet", "center" and "xent_sc" for for classification of object classes
    - num_query (int): Number of queries in dataset of data_loader
    - start_epoch (int): Initial state.epoch parameter of training Engine.
    """
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline")
    logger.info("Start training")
    web_logger = WebLogger(epochs)

    if cfg.WANDB.LOG_WANDB == 'on':
        setup_wandb(cfg, output_dir)

    trainer = create_supervised_trainer(model, optimizer, criterion, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)

    if cfg.TEST.PARTIAL_REID == 'off':
        evaluator = create_supervised_evaluator(
            model,
            metrics={
                'r1_mAP_mINP': r1_mAP_mINP(
                    num_query,
                    max_rank=50,
                    feat_norm=cfg.TEST.FEAT_NORM,
                    save_memory=cfg.TEST.SAVE_MEMORY,
                    block_size=cfg.TEST.SAVE_MEMORY_BLOCK,
                    logger=logger
                )
            },
            device=device
        )
    else:
        evaluator_reid = create_supervised_evaluator(
            model,
            metrics={'r1_mAP_mINP': r1_mAP_mINP(300, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
            device=device
        )
        evaluator_ilids = create_supervised_evaluator(
            model,
            metrics={'r1_mAP_mINP': r1_mAP_mINP(119, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
            device=device
        )
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=None, require_empty=False)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=checkpoint_period),
        checkpointer,
        {
            'model': model,
            'optimizer': optimizer['model'],
            'center_param': criterion['center'],
            'optimizer_center': optimizer['center']
        }
    )
    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED
    )

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.EPOCH_STARTED)
    def weblog_epoch_start(engine):
        if cfg.TEST.PARTIAL_REID == 'off':
            if eval_period == 1 or (engine.state.epoch + eval_period - 1 <= epochs and engine.state.epoch % eval_period == 1):
                web_logger.on_step_begin()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                engine.state.epoch,
                ITER,
                len(data_loader['train']),
                engine.state.metrics['avg_loss'],
                engine.state.metrics['avg_acc'],
                scheduler.get_lr()[0]
            ))
            if cfg.WANDB.LOG_WANDB == 'on':
                wandb.log({
                    "epoch": engine.state.epoch,
                    "acc": engine.state.metrics['avg_acc'],
                    "loss": engine.state.metrics['avg_loss'],
                    "lr": scheduler.get_lr()[0]
                })
        if len(data_loader['train']) == ITER:
            ITER = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
            engine.state.epoch,
            timer.value() * timer.step_count,
            data_loader['train'].batch_size / timer.value()
        ))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            if cfg.TEST.PARTIAL_REID == 'off':
                evaluator.run(data_loader['eval'])
                cmc, mAP, mINP, tabular_data = evaluator.state.metrics['r1_mAP_mINP']
                web_metrics = {}
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                logger.info("mINP: {:.1%}".format(mINP))
                logger.info("mAP: {:.1%}".format(mAP))
                web_metrics["mAP"] = mAP
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    web_metrics[f"CMC-{r}"] = cmc[r - 1]
                web_logger.on_step_end(engine.state.epoch-1, web_metrics)
                if cfg.WANDB.LOG_WANDB == 'on':
                    log_eval_wandb(mAP, cmc, tabular_data)
                logger.info("----------")
            else:
                evaluator_reid.run(data_loader['eval_reid'])
                cmc, mAP, mINP = evaluator_reid.state.metrics['r1_mAP_mINP']
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                logger.info("mINP: {:.1%}".format(mINP))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 3, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                evaluator_ilids.run(data_loader['eval_ilids'])
                cmc, mAP, mINP = evaluator_ilids.state.metrics['r1_mAP_mINP']
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                logger.info("mINP: {:.1%}".format(mINP))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 3, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_eval_progress(engine):
        global ITER
        ITER += 1
        len_max = len(data_loader["eval"])

        if ITER % log_period == 0:
            logger.info(f"Evaluation Data Processing {ITER}/{len_max} done.")
        if len(data_loader['eval']) == ITER:
            ITER = 0

    trainer.run(data_loader['train'], max_epochs=epochs)
