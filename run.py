import gc
import os

import torch

import wandb
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.config import get_args


def get_exp(args):
    if args.task_name == "long_term_forecast":
        return Exp_Long_Term_Forecast(args)
    if args.task_name == "short_term_forecast":
        return Exp_Short_Term_Forecast(args)
    if args.task_name == "imputation":
        return Exp_Imputation(args)
    if args.task_name == "anomaly_detection":
        return Exp_Anomaly_Detection(args)
    if args.task_name == "classification":
        return Exp_Classification(args)
    return Exp_Long_Term_Forecast(args)


def clean(ckpt=None):
    """
    Clean up the cache and remove checkpoints.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
        parent_dir = os.path.dirname(ckpt)
        if not os.listdir(parent_dir):
            os.rmdir(parent_dir)


def get_setting(args):
    setting = f"{args.task_name}_{args.model_id}_{args.model}_attndropout{args.attn_dropout}_hiddendim{args.hidden_dim}_mlp-ratio{args.mlp_ratio}_n-depth{args.n_depth}_n-emb{args.n_emb}_n-heads{args.n_heads}"

    if "cndiff" in args.model.lower():
        setting += f"_timesteps{args.timesteps}_{args.des}_use_cond{args.use_cond}_use_tphi{args.use_tphi}_normalize{args.normalize}_tphi-loss{args.tphi_loss}"

    if args.task_name == "classification":
        setting += f"_classifier{args.classifier}"

    return setting


if __name__ == "__main__":
    args = get_args()
    setting = get_setting(args)

    import json

    filename = f"results/{args.filename or args.task_name}_results.json"
    if os.path.exists(filename) and args.task_name != "classification":
        resultdict = json.load(open(filename))
        if setting in resultdict:
            print(f"{args.task_name} Experiment {setting} already exists. Exiting.")
            exit(0)

    if args.wandb:
        wandb.init(
            project="CnDiff",
            name=setting,
            config=vars(args),
            reinit=True,
        )
    exp = get_exp(args)
    # print(exp.args)

    try:
        if args.is_training:
            print(f">>>>>>>TRAINING : \n{setting}\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.train(setting)

        print(f">>>>>>>TESTING : \n{setting}\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        ckpt = exp.test(setting)
        clean(ckpt)

        print("-------------------------------------------------")
    except torch.cuda.OutOfMemoryError:
        print(">>>>>>>>>>> OOM Error <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
