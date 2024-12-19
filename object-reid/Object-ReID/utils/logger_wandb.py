import wandb
#from nicr_cluster_utils.utils.wandb_integration import is_wandb_available  # proprietary
from utils.cfg_to_dict import cfg_to_dict


def setup_wandb(cfg, output_dir):
    """
    Setup Weights and Biases API from config.

    Args:
    - cfg (CfgNode): Config of experiment
    - output_dir (str): Ouput directory for wandb log
    """
    cfg_dict = cfg_to_dict(cfg)
    cfg_wandb = cfg_dict["WANDB"]
    for k, v in cfg_wandb.items():
        if v == "":
            cfg_wandb[k] = None
    mode = "online"# if is_wandb_available() else "offline"
    wandb.init(
        project=cfg_wandb["PROJECT"],
        notes=cfg_wandb["NOTES"],
        tags=cfg_wandb["TAGS"],
        config=cfg_dict,
        dir=output_dir,
        mode=mode
    )


def log_eval_wandb(mAP, cmc, tabular_data):
    """
    Log evaluation results to wandb project. A bunch of scalar metrics are logged with the run,
    a lot of additional metrics as an artifact under "query_data".

    Args:
    - mAP (float): mAP results
    - cmc (array): CMC array with entries from 1 to k
    - tabular_data (DataFrame): DataFrame with lots of additional metrics for each indiviudal query
    """
    log_item = {
        "mAP": mAP
    }
    for r in [1, 5, 10]:
        log_item[f"CMC-{r}"] = cmc[r - 1]
    tabular_data_grp = tabular_data.groupby("class")
    for attr in tabular_data.drop(columns=["id", "class"]).columns:
        log_item[f"mean_{attr}"] = tabular_data[attr].mean()
        log_item[f"mean_class_{attr}"] = tabular_data_grp[attr].mean().mean()
    wandb.log(log_item, commit=False)
    table = wandb.Table(data=tabular_data)
    query_data = wandb.Artifact(name="Eval_" + str(wandb.run.name), type="query_data")
    query_data.add(table, "query_table")
    wandb.run.log_artifact(query_data)
