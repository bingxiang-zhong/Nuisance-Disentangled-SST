import math
import os
import sys


from baseline_config import build_sim_data_params, parse_args
from datasets.noise_dataset import NoiseDataset
from tqdm import tqdm
from utils import save_file, set_seed, Parameter

def _print_and_flush(msg):
    print(msg)
    sys.stdout.flush()


def _resolve_source_path(params):
    data_type = params["data_type"]
    path_key = f"path_{data_type}"
    source_path = params.get(path_key)

    if source_path is None:
        raise ValueError(f"Missing source path for data_type={data_type}: {path_key}")

    return source_path


def _resolve_save_dir(params):
    save_path = params["save_path"]
    data_type = params["data_type"]

    if os.path.basename(os.path.normpath(save_path)) == data_type:
        return save_path

    return os.path.join(save_path, data_type)


def _build_random_trajectory_dataset(source_signal_dataset, noise_dataset, params):
    dataset_params = params["dataset"]
    max_rt60 = dataset_params["max_rt60"]

    from datasets.random_trajectory_dataset import RandomTrajectoryDataset

    return RandomTrajectoryDataset(
        sourceDataset=source_signal_dataset,
        noiseDataset=noise_dataset,
        room_sz=Parameter([3, 3, 2.5], [10, 8, 6]),
        T60=Parameter(0.2, max_rt60) if max_rt60 > 0 else 0,
        abs_weights=Parameter([0.5] * 6, [1.0] * 6),
        array=dataset_params.get("array", "realman"),
        array_pos=Parameter([0.1, 0.1, 0.3], [0.9, 0.5, 0.5]),
        SNR=Parameter(-5, 15),
        nb_points=50,
        noise_type=dataset_params.get("noise_type", "omni"),
        win_size=params["win_size"],
        hop_rate=params["hop_rate"],
    )


def _save_generated_dataset(dataset, save_dir, data_size, batch_size):
    if data_size <= 0:
        raise ValueError(f"data_size must be positive, got {data_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    os.makedirs(save_dir, exist_ok=True)
    _print_and_flush(f"Saving generated data to: {save_dir}")
    _print_and_flush(f"Generating {data_size} samples with batch_size={batch_size}")

    batch_num = math.ceil(data_size / batch_size)
    sample_idx = 0

    for batch_idx in tqdm(range(batch_num)):
        current_batch_size = min(batch_size, data_size - sample_idx)
        idx1 = batch_size * batch_idx
        idx2 = idx1 + current_batch_size

        mic_signals_batch, acoustic_scene_batch = dataset.get_batch(idx1, idx2)

        for i in range(current_batch_size):
            sig_path_b = os.path.join(save_dir, f"{sample_idx}.wav")
            acous_path = os.path.join(save_dir, f"{sample_idx}.npz")

            save_file(
                mic_signals_batch[i],
                acoustic_scene_batch[i],
                sig_path_b,
                acous_path,
            )
            sample_idx += 1


def main():
    args = parse_args()
    params = build_sim_data_params(args)

    seed = params["seed"]
    set_seed(seed)

    _print_and_flush(f"Data generation parameters: {params}")

    T = params["dataset"]["max_audio_len_s"]
    source_path = _resolve_source_path(params)
    save_dir = _resolve_save_dir(params)
    batch_size = params["training"]["batch_size"]
    data_size = params["data_size"]

    # Avoid loading gpuRIR if not needed, so that the code can be tested on a machine without a GPU
    from datasets.librispeech_dataset import LibriSpeechDataset

    source_signal_dataset = LibriSpeechDataset(source_path, T, return_vad=True)

    noise_dataset = NoiseDataset(
        T=T,
        fs=params["fs"],
        nmic=params["mic_num"],
        noise_type='diffuse',
        noise_path=params["noise_path"])

    # %%
    dataset = _build_random_trajectory_dataset(
        source_signal_dataset=source_signal_dataset,
        noise_dataset=noise_dataset,
        params=params,
    )

    _save_generated_dataset(dataset, save_dir, data_size, batch_size)

    print("data generation finished")


if __name__ == "__main__":
    
    main()
