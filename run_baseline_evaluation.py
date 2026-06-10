import sys
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_config import build_baseline_params, build_parser
from datasets.array_setup import ARRAY_SETUPS
from datasets.fix_trajectory_dataset import FixTrajectoryDataset
from trainers.crnn_trainer import CRNNTrainer
from utils import build_real_loader, collate_fn, set_seed


DEFAULT_CHECKPOINT = "checkpoints/base.bin"
DEFAULT_TEST_DIR = os.environ.get("SSL_TEST_DIR", os.path.join("data", "generated", "test"))
DEFAULT_REAL_ENV = "OfficeRoom3"
DEFAULT_REAL_SNRS = [-10, -5, 0, 5, 10, 15]


def build_eval_parser():
    parser = build_parser()
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--realman_root",
        type=str,
        default=os.environ.get("REALMAN_ROOT", os.path.join("data", "RealMAN")),
    )
    parser.add_argument("--real_env", type=str, default=DEFAULT_REAL_ENV)
    parser.add_argument("--real_snr", type=int, nargs="+", default=DEFAULT_REAL_SNRS)
    return parser


def parse_eval_args(argv=None):
    return build_eval_parser().parse_args(argv)


def build_params(args):
    params = build_baseline_params(args)
    params["model_checkpoint_path"] = ""
    params["mic_num"] = ARRAY_SETUPS["realman"]["mic_pos"].shape[0]

    if params.get("norm_type") is None:
        params["norm_type"] = "gn"

    return params


def build_trainer(params, checkpoint_path):
    if params["model"].lower() != "crnn":
        raise ValueError(f"run_baseline_evaluation.py only supports crnn, got: {params['model']}")

    trainer = CRNNTrainer(params, training=False)
    trainer.load_checkpoint(checkpoint_path)

    if torch.cuda.is_available():
        trainer.cuda()

    return trainer


def build_synthetic_loader(test_dir, batch_size):
    dataset = FixTrajectoryDataset(test_dir, dataset_sz=None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )


def print_metrics(prefix, metrics):
    print(
        prefix,
        "MAE", float(metrics["MAE"]),
        "ACC", float(metrics["ACC"]),
        "loss", float(metrics["loss"]),
    )


def main():
    args = parse_eval_args()
    checkpoint_path = args.checkpoint_path
    params = build_params(args)
    seed = params["training"].get("seed", 42)
    set_seed(seed)

    batch_size = params["training"].get("batch_size", 32)
    test_dir = args.test_dir

    print(f"Loading checkpoint: {checkpoint_path}")
    print("Using baseline_config and CLI params.")
    print(f"Evaluating synthetic dataset: {test_dir}")
    sys.stdout.flush()

    trainer = build_trainer(params, checkpoint_path)
    synthetic_loader = build_synthetic_loader(test_dir, batch_size)
    synthetic_metrics = trainer.test_epoch(synthetic_loader)
    print_metrics("synthetic", synthetic_metrics)

    print(f"Evaluating RealMAN environment: {args.real_env}")
    real_loaders = build_real_loader(
        batch_size=batch_size,
        environment=args.real_env,
        realman_root=args.realman_root,
        snr_list=args.real_snr,
        seed=seed,
    )
    real_metrics = {}
    for snr, loader in real_loaders.items():
        metrics = trainer.test_epoch(loader)
        real_metrics[snr] = metrics
        print_metrics(f"real_snr_{snr}", metrics)

    if real_metrics:
        print(
            "real_average",
            "MAE", float(np.mean([m["MAE"] for m in real_metrics.values()])),
            "ACC", float(np.mean([m["ACC"] for m in real_metrics.values()])),
            "loss", float(np.mean([m["loss"] for m in real_metrics.values()])),
        )


if __name__ == "__main__":
    main()
