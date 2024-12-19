# encoding: utf-8
import logging
import torch
from ignite.engine import Engine, Events

from utils.reid_metric import r1_mAP_mINP
from survey.utils.reid_metric import r1_mAP_mINP_reranking
from utils.logger_wandb import setup_wandb, log_eval_wandb

global ITER
ITER = 0


def create_supervised_evaluator(model, metrics, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to evaluate
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, other = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids, other

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_test(
        cfg,
        output_dir,
        model,
        data_loader,
        num_query
):
    """
    Do evaluation on dataset and log results.

    Args:
    - cfg (CfgNode): Config of experiment
    - output_dir (str): Path to output directory
    - model (Module): Loaded torch model
    - data_loader (dict): Dict containing DataLoader at key "eval"
    - num_query (int): Number of queries in dataset of data_loader
    """
    device = cfg.MODEL.DEVICE
    logger = logging.getLogger("reid_baseline")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'off':
        print("Create evaluator")
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
    elif cfg.TEST.RE_RANKING == 'on':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP_mINP': r1_mAP_mINP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for on or off, but got {}.".format(cfg.TEST.RE_RANKING))

    if cfg.WANDB.LOG_WANDB == 'on':
        setup_wandb(cfg, output_dir)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_eval_progress(engine):
        global ITER
        ITER += 1
        len_max = len(data_loader["eval"])

        if ITER % 20 == 0:
            logger.info(f"Evaluation Data Processing {ITER}/{len_max} done.")
        if len(data_loader['eval']) == ITER:
            ITER = 0

    evaluator.run(data_loader['eval'])
    cmc, mAP, mINP, tabular_data = evaluator.state.metrics['r1_mAP_mINP']
    logger.info('Validation Results')
    logger.info("mINP: {:.1%}".format(mINP))
    logger.info("mAP: {:.1%}".format(mAP))
    if cfg.TEST.PARTIAL_REID == 'off':
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        if cfg.WANDB.LOG_WANDB == 'on':
            log_eval_wandb(mAP, cmc, tabular_data)
    else:
        for r in [1, 3, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
