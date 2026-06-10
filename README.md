# Nuisance-Disentangled-SST

Source code for the EUSIPCO 2026 paper **"Domain-Robust Sound Source Tracking Using Nuisance Disentanglement"**.

The project provides a compact CRNN-based pipeline for sound source localization and tracking with simulated microphone-array recordings. It includes utilities for generating synthetic trajectories, training a baseline model, training the proposed nuisance-disentangled variant, and evaluating on synthetic and RealMAN-style data.

## Repository Layout

- `run_baseline.py`: train the CRNN model with `base` or `disentangled` training mode.
- `run_baseline_evaluation.py`: evaluate a trained checkpoint on synthetic data and optional RealMAN data.
- `simData.py`: generate simulated microphone-array data from LibriSpeech-style speech and NoiseX-92 noise.
- `baseline_config.py`: default configuration and CLI overrides.
- `datasets/`: dataset loaders, array geometry definitions, noise generation, and trajectory simulation.
- `models/`: CRNN model and signal-processing transforms.
- `trainers/`: training and evaluation loops.
- `utils.py`: shared utilities for seeding, collation, I/O, schedulers, and RealMAN loaders.

## Data And Artifacts

Large generated data, logs, and checkpoints are intentionally excluded from Git. By default the code expects local data under:

```text
data/
  LibriSpeech/
    train-clean-100/
    test-clean/
  generated/
    train/
    val/
    test/
  RealMAN/
datasets/
  NoiseX-92/
```

You can override paths with CLI arguments or environment variables:

```bash
export SSL_DATA_ROOT=/path/to/data
export LIBRISPEECH_TRAIN_DIR=/path/to/LibriSpeech/train-clean-100
export LIBRISPEECH_TEST_DIR=/path/to/LibriSpeech/test-clean
export NOISEX92_DIR=/path/to/NoiseX-92
export SSL_TRAIN_DIR=/path/to/generated/train
export SSL_VAL_DIR=/path/to/generated/val
export SSL_TEST_DIR=/path/to/generated/test
export REALMAN_ROOT=/path/to/RealMAN
```

## Installation

Create a Python environment and install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`gpuRIR` requires a compatible CUDA runtime/driver setup. If CUDA is not available, data simulation may fail, but training and evaluation on pre-generated data can still work if the required data files are present.

## Generate Synthetic Data

Generate training data:

```bash
python simData.py \
  --data_type train \
  --save_path data/generated \
  --data_size 160000 \
  --batch_size 32 \
  --path_train "$LIBRISPEECH_TRAIN_DIR" \
  --noise_path "$NOISEX92_DIR"
```

Generate validation or test data by changing `--data_type` and the source path:

```bash
python simData.py \
  --data_type test \
  --save_path data/generated \
  --data_size 1600 \
  --path_test "$LIBRISPEECH_TEST_DIR"
```

## Train

Train the proposed disentangled CRNN variant:

```bash
python run_baseline.py \
  --model crnn \
  --training_mode disentangled \
  --path_train data/generated/train \
  --path_val data/generated/val \
  --path_test data/generated/test \
  --batch_size 64 \
  --num_workers 4 \
  --nb_epochs 30 \
  --seed 42 \
  --lambda_mi 0.00001
```

For a plain baseline run, use `--training_mode base` and omit `--lambda_mi`.

Training outputs are written under `logs/` and are ignored by Git.

## Evaluate

Evaluate a checkpoint on synthetic data:

```bash
python run_baseline_evaluation.py \
  --checkpoint_path checkpoints/disentangled_best_ep.bin \
  --training_mode disentangled \
  --test_dir data/generated/test \
  --batch_size 32
```

To include RealMAN evaluation, provide the dataset root:

```bash
python run_baseline_evaluation.py \
  --checkpoint_path checkpoints/disentangled.bin \
  --training_mode disentangled \
  --test_dir data/generated/test \
  --realman_root data/RealMAN \
  --real_env OfficeRoom3 \
  --real_snr -10 -5 0 5 10 15
```

Use `--checkpoint_path checkpoints/base.bin --training_mode base` when evaluating a baseline checkpoint.

### RealMAN Data Layout

`run_baseline_evaluation.py` calls `utils.build_real_loader()` to create one `DataLoader` per requested SNR. Internally, this wraps `datasets.RealRecord.RealData`.

The loader expects `--realman_root` to point to a RealMAN-style dataset directory. `RealData` first reads the target CSV and uses its `filename` column to locate audio files under `realman_root`:

```text
RealMAN/
  test/
    test_moving_source_location.csv
    ma_speech/
        OfficeRoom3/
        ...
    dp_speech/
        OfficeRoom3/
        ...
    ma_noisy_speech/
        OfficeRoom3/
        ...
    ma_noise/
        OfficeRoom3/
          ...
```

The important convention is that each row in `test_moving_source_location.csv` contains a `filename` relative to `realman_root`, typically pointing to a `ma_noisy_speech/.../*_CH1.flac` file. `RealData` then derives the actual files it needs from that path:

- Target angles are read from the CSV `angle(°)` column.
- The CSV `filename` column is joined with `realman_root` to build the original path.
- Clean microphone signals are loaded by replacing `ma_noisy_speech` with `ma_speech`.
- Direct-path speech for VAD is loaded by replacing `ma_noisy_speech` with `dp_speech`.
- Multi-channel files are loaded by replacing the channel suffix with `_CH<i>.flac`.
- Noise files are searched under `test/ma_noise/<real_env>/`.
- `--real_snr` controls the SNR range passed to `RealData`; the evaluation script builds one loader per requested SNR value.

In other words, this command:

```bash
python run_baseline_evaluation.py \
  --realman_root data/RealMAN \
  --real_env OfficeRoom3 \
  --real_snr -10 -5 0 5 10 15
```

loads annotations from `data/RealMAN/test/test_moving_source_location.csv`, filters CSV rows whose filenames contain `OfficeRoom3`, loads matching speech/direct-path files using the naming convention above, samples noise from `data/RealMAN/test/ma_noise/OfficeRoom3/`, and reports metrics separately for SNR -10, -5, 0, 5, 10, and 15 before averaging them.

## References

- [Cross3D](https://github.com/DavidDiazGuerra/Cross3D)
- [Neural-SRP](https://github.com/egrinstein/neural_srp)
- [RealMAN dataset](https://github.com/Audio-WestlakeU/RealMAN)

## License

This project is released under the MIT License. See `LICENSE` for details.
