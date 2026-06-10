
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from datetime import datetime
from datasets.array_setup import ARRAY_SETUPS
from trainers.crnn_trainer import CRNNTrainer
from datasets.fix_trajectory_dataset import  FixTrajectoryDataset
from utils import Parameter, set_seed, EarlyStopping, seed_worker, make_generator, collate_fn
import random
import numpy as np

import math
from baseline_config import build_baseline_params, parse_args


def _print_and_flush(msg):
    print(msg)
    sys.stdout.flush()


def build_trainer(params):
    trainer_map = {
        "crnn": CRNNTrainer,
    }

    model_name = params["model"].lower()
    if model_name not in trainer_map:
        raise ValueError(f"Unsupported model: {params['model']}")

    return trainer_map[model_name](params)


def train_model(params, train_loader, val_loader, test_loader, dir_name):

    nb_epoch = params["training"]["nb_epochs"]

    params["len_train_loader"] = len(train_loader)
    ds = params["data_size"]

    model_name = params["model"]
    trainer = build_trainer(params)

    if torch.cuda.is_available():
        trainer.cuda()

    print("Training network...")

    start_time_str = datetime.now().strftime("%m-%d_%Hh%Mm")
    run_dir = os.path.join(dir_name, f"run_{start_time_str}_ds_{ds}")
    os.makedirs(run_dir, exist_ok=True)

    log_file = f"{run_dir}/training_metrics.jsonl"

    with open(os.path.join(run_dir, "params.json"), "w") as json_file:
        json.dump(params, json_file, indent=4)

    syn_monitor_name = "loss"


    syn_early_stopper = EarlyStopping(
        patience=8,
        min_delta=0.0,
        mode="min",
    )

    best_syn_score = float("inf")
    best_syn_epoch = None
    best_syn_model_path = None

    def write_jsonl(row):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def remove_file_if_exists(path):
        if path is not None and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                _print_and_flush(f"Warning: failed to remove old checkpoint: {e}")

    def evaluate_test_loader():
        metrics = trainer.test_epoch(test_loader)

        return {
            "test_loss": float(metrics["loss"]),
            "test_MAE": float(metrics["MAE"]),
            "test_ACC": float(metrics["ACC"]),
        }

    for epoch_idx in range(1, nb_epoch + 1):

        _print_and_flush(f"\nEpoch {epoch_idx}/{nb_epoch}:")

        trainer.train_epoch(
            train_loader,
            epoch=epoch_idx,
        )

        syn_metrics = trainer.test_epoch(val_loader)
        cur_lr = trainer.optimizer.param_groups[0]["lr"]

        epoch_metric = {
            "epoch": epoch_idx,
         
            "syn_monitor_name": syn_monitor_name,
            "syn_loss": float(syn_metrics["loss"]),
            "syn_MAE": float(syn_metrics["MAE"]),
            "syn_ACC": float(syn_metrics["ACC"]),

            "timestamp": datetime.now().isoformat(),
        }

        write_jsonl(epoch_metric)

        _print_and_flush(f"Current LR: {cur_lr:.6e}")

        _print_and_flush(
            "Syn loss: {:.4f}, "
            "Syn mae azi: {:.2f}deg, "
            "Syn acc: {:.2f}%".format(
                epoch_metric["syn_loss"],
                epoch_metric["syn_MAE"],
                epoch_metric["syn_ACC"],
            )
        )


        # =========================================================
        # Syn checkpoint selection + early stopping
        # =========================================================
        current_syn_score = epoch_metric[f"syn_{syn_monitor_name}"]

        syn_improved = syn_early_stopper.step(
            score=current_syn_score,
            epoch=epoch_idx,
            trainer=trainer,
            run_dir=run_dir,
            save_prefix="best_syn",
        )

        if syn_improved:
            _print_and_flush(
                f"New best SYN model found at epoch {epoch_idx}, "
                f"{syn_monitor_name} = {current_syn_score:.6f}"
            )
        else:
            _print_and_flush(
                f"No improvement in syn {syn_monitor_name}. "
                f"Early stop counter: "
                f"{syn_early_stopper.counter}/{syn_early_stopper.patience}"
            )

        if syn_early_stopper.should_stop:
            _print_and_flush(
                f"Early stopping triggered at epoch {epoch_idx}. "
                f"Best SYN epoch: {syn_early_stopper.best_epoch}, "
                f"best SYN {syn_monitor_name}: {syn_early_stopper.best_score:.6f}"
            )
            break

    best_syn_epoch = syn_early_stopper.best_epoch
    best_syn_score = syn_early_stopper.best_score
    best_syn_model_path = syn_early_stopper.best_model_path

    _print_and_flush(
        f"Best SYN epoch: {best_syn_epoch}, "
        f"best {syn_monitor_name}: {best_syn_score:.6f}"
    )


    # =========================================================
    # Final evaluation: best SYN model
    # No val_loader evaluation here.
    # =========================================================
    if best_syn_model_path is not None and os.path.exists(best_syn_model_path):
        trainer.load_checkpoint(best_syn_model_path)

        best_syn_test_metrics = evaluate_test_loader()

        best_syn_summary = {
            "selection_type": "best_syn_model",
            "epoch": best_syn_epoch,

            "syn_monitor_name": syn_monitor_name,
            "best_syn_epoch": best_syn_epoch,
            "best_syn_score": float(best_syn_score),
            "best_syn_model_path": best_syn_model_path,

            "test_loss": best_syn_test_metrics["test_loss"],
            "test_MAE": best_syn_test_metrics["test_MAE"],
            "test_ACC": best_syn_test_metrics["test_ACC"],

            "timestamp": datetime.now().isoformat(),
        }

        write_jsonl(best_syn_summary)

        _print_and_flush("\nBest SYN model evaluation:")
        _print_and_flush(
            "Test loss: {:.4f}, "
            "Test mae azi: {:.2f}deg, "
            "Test acc: {:.2f}%".format(
                best_syn_summary["test_loss"],
                best_syn_summary["test_MAE"],
                best_syn_summary["test_ACC"],
            )
        )

    else:
        _print_and_flush("Warning: no best SYN model was saved.")


    _print_and_flush("Training finished.")


def main():
    
    args = parse_args()

    params = build_baseline_params(args)
    model_name = params["model"]
    batch_size = params["training"]["batch_size"]
    num_workers = params["training"]["num_workers"]
    seed = params["training"]["seed"]
    training_mode = params["training_mode"]
  
    mic_num = ARRAY_SETUPS["realman"]["mic_pos"].shape[0]
 
    params["mic_num"] = mic_num
    print(f"Using {mic_num}-channel input based on the specified array setup.")


    run_dir = os.path.join(
        "logs",
        f"model_{model_name}",
        f"mode_{training_mode}",
        f"seed_{seed}",
    )

    train_dataset_path = params["path_train"]
    val_dataset_path = params["path_val"]
    test_dataset_path = params["path_test"]

    set_seed(seed)
    data_size = params["data_size"]
    dataset_train = FixTrajectoryDataset(train_dataset_path, dataset_sz=data_size)
    if data_size is None:
        params["data_size"] = len(dataset_train)
    print("Training dataset length: ", len(dataset_train))
    dataset_test = FixTrajectoryDataset(test_dataset_path, dataset_sz=None)
    dataset_val = FixTrajectoryDataset(val_dataset_path, dataset_sz=None)


    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    train_loader = DataLoader(dataset_train, shuffle=True, worker_init_fn=seed_worker,
        generator=make_generator(seed),
        **loader_kwargs,)

    val_loader = DataLoader(dataset_val, shuffle=False, **loader_kwargs,)
    test_loader = DataLoader(dataset_test, shuffle=False, **loader_kwargs)
    

    
    train_model(params, train_loader, val_loader, test_loader, run_dir)
       
if __name__ == '__main__':
    main()
