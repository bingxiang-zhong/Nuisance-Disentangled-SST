import argparse
import os
from copy import deepcopy


DEFAULT_DATA_SIZE = None
DEFAULT_SIM_DATA_SIZE = 16000

DATA_ROOT = os.environ.get("SSL_DATA_ROOT", "data")

DEFAULT_PATH_TRAIN = os.environ.get(
    "LIBRISPEECH_TRAIN_DIR",
    os.path.join(DATA_ROOT, "LibriSpeech", "train-clean-100"),
)
DEFAULT_PATH_TEST = os.environ.get(
    "LIBRISPEECH_TEST_DIR",
    os.path.join(DATA_ROOT, "LibriSpeech", "test-clean"),
)
DEFAULT_NOISE_PATH = os.environ.get(
    "NOISEX92_DIR",
    os.path.join("datasets", "NoiseX-92"),
)
DEFAULT_BASELINE_PATH_TRAIN = os.environ.get(
    "SSL_TRAIN_DIR",
    os.path.join(DATA_ROOT, "generated", "train"),
)
DEFAULT_BASELINE_PATH_VAL = os.environ.get(
    "SSL_VAL_DIR",
    os.path.join(DATA_ROOT, "generated", "val"),
)
DEFAULT_BASELINE_PATH_TEST = os.environ.get(
    "SSL_TEST_DIR",
    os.path.join(DATA_ROOT, "generated", "test"),
)

SIM_DATA_DEFAULTS = {
    "path_train": DEFAULT_PATH_TRAIN,
    "path_val": None,
    "path_test": DEFAULT_PATH_TEST,
    "save_path": "generated_data",
    "data_type": "train",
    "data_size": DEFAULT_SIM_DATA_SIZE,
    "seed": 101,
    "fs": 16000,
    "mic_num": 5,
    "win_size": 1600,
    "hop_rate": 1,
    "noise_path": DEFAULT_NOISE_PATH,
    "training": {
        "batch_size": 32,
    },
    "dataset": {
        "max_audio_len_s": 5,
        "max_rt60": 1.0,
        "array": "realman",
        "noise_type": "omni",
    },
}

BASELINE_DEFAULTS = {
    "model": "crnn",
    "training_mode": "base",
    "path_train": DEFAULT_BASELINE_PATH_TRAIN,
    "path_val": DEFAULT_BASELINE_PATH_VAL,
    "path_test": DEFAULT_BASELINE_PATH_TEST,
    "data_size": DEFAULT_DATA_SIZE,
    "sigma": 16,
    "norm_type": "gn",
    "fs": 16000,
    "speed_of_sound": 343,
    "win_size": 512,
    "hop_rate": 0.625,
    "res_phi": 360,
    "warm_up_ratio": 0.05,
    "min_lr_ratio": 10,
    "training": {
        "batch_size": 64,
        "num_workers": 4,
        "lr": 0.0005,
        "nb_epochs": 30,
        "seed": 42,
    },
}

CLI_ARGUMENTS = (
    (("--model",), {"type": str, "default": None}),
    (("--batch_size",), {"type": int, "default": None}),
    (("--num_workers",), {"type": int, "default": None}),
    (("--nb_epochs",), {"type": int, "default": None}),
    (("--data_size",), {"type": int, "default": None}),
    (("--lr",), {"type": float, "default": None}),
    (("--seed",), {"type": int, "default": None}),
    (("--training_mode",), {"type": str, "default": None}),
    (("--lambda_mi",), {"type": float, "default": None}),
    (
        ("--use_amp",),
        {
            "action": "store_true",
            "help": "Use automatic mixed precision (AMP) during training.",
        },
    ),
    (("--warm_up_ratio",), {"type": float, "default": 0.05}),
    (("--min_lr_ratio",), {"type": float, "default": 10}),
    (("--sigma",), {"type": float, "default": 16}),
    (("--res_phi",), {"type": int, "default": None}),
    (("--norm_type",), {"type": str, "default": None}),
    (("--path_train",), {"type": str, "default": None}),
    (("--path_val",), {"type": str, "default": None}),
    (("--path_test",), {"type": str, "default": None}),
    (("--save_path",), {"type": str, "default": None}),
    (("--data_type",), {"type": str, "choices": ("train", "val", "test"), "default": None}),
    (("--noise_path",), {"type": str, "default": None}),
    (("--max_audio_len_s",), {"type": int, "default": None}),
    (("--max_rt60",), {"type": float, "default": None}),
    (("--array",), {"type": str, "default": None}),
    (("--noise_type",), {"type": str, "default": None}),
)


PARAM_OVERRIDES = {
    "model": ("model",),
    "batch_size": ("training", "batch_size"),
    "num_workers": ("training", "num_workers"),
    "nb_epochs": ("training", "nb_epochs"),
    "data_size": ("data_size",),
    "lr": ("training", "lr"),
    "seed": ("training", "seed"),
    "training_mode": ("training_mode",),
    "lambda_mi": ("training", "lambda_mi"),
    "use_amp": ("training", "use_amp"),
    "warm_up_ratio": ("warm_up_ratio",),
    "min_lr_ratio": ("min_lr_ratio",),
    "sigma": ("sigma",),
    "res_phi": ("res_phi",),
    "norm_type": ("norm_type",),
    "path_train": ("path_train",),
    "path_val": ("path_val",),
    "path_test": ("path_test",),
    "save_path": ("save_path",),
    "data_type": ("data_type",),
    "noise_path": ("noise_path",),
    "max_audio_len_s": ("dataset", "max_audio_len_s"),
    "max_rt60": ("dataset", "max_rt60"),
    "array": ("dataset", "array"),
    "noise_type": ("dataset", "noise_type"),
}

SIM_DATA_PARAM_OVERRIDES = {
    "batch_size": ("training", "batch_size"),
    "data_size": ("data_size",),
    "seed": ("seed",),
    "path_train": ("path_train",),
    "path_val": ("path_val",),
    "path_test": ("path_test",),
    "save_path": ("save_path",),
    "data_type": ("data_type",),
    "noise_path": ("noise_path",),
    "max_audio_len_s": ("dataset", "max_audio_len_s"),
    "max_rt60": ("dataset", "max_rt60"),
    "array": ("dataset", "array"),
    "noise_type": ("dataset", "noise_type"),
}


def build_parser():
    parser = argparse.ArgumentParser()
    for flags, kwargs in CLI_ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def merge_args_into_params(params, args):
    merged = deepcopy(params)

    for arg_name, param_path in PARAM_OVERRIDES.items():
        value = getattr(args, arg_name)
        if value is not None:
            _set_nested_value(merged, param_path, value)

    return merged


def build_baseline_params(args):
    return merge_args_into_params(BASELINE_DEFAULTS, args)


def build_sim_data_params(args):
    params = deepcopy(SIM_DATA_DEFAULTS)

    for arg_name, param_path in SIM_DATA_PARAM_OVERRIDES.items():
        value = getattr(args, arg_name)
        if value is not None:
            _set_nested_value(params, param_path, value)

    data_type = params["data_type"]
    if data_type == "val" and params.get("path_val") is None:
        params["path_val"] = params["path_test"]

    return params


def _set_nested_value(params, path, value):
    cursor = params
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value
