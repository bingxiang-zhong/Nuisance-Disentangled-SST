
import math
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
import soundfile
import random
import pickle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def collate_fn(batch):
    """
    batch: list of (mic, meta)
      mic  -> numpy array
      meta -> dict with keys: "DOAw", "vad"
    """
    from datasets.array_setup import ARRAY_SETUPS

    mics, metas = zip(*batch)
    gt_batch = {}

    mic_batch = torch.as_tensor(
        np.stack(mics),
        dtype=torch.float32,
    )

    doa_batch = torch.as_tensor(
        np.stack([m["DOAw"] for m in metas]),
        dtype=torch.long,
    )

    if "vad" in metas[0]:
        vad_np = np.stack([m["vad"] for m in metas])
        vad_np = vad_np.mean(axis=2) > (2.0 / 3.0)
        vad_batch = torch.as_tensor(vad_np, dtype=torch.bool).unsqueeze(-1)
    else:
        vad_batch = torch.ones(doa_batch.shape[:2], dtype=torch.bool).unsqueeze(-1)

    gt_batch["doa"] = doa_batch
    gt_batch["vad"] = vad_batch
    gt_batch["mic_pos"] = torch.as_tensor(
        ARRAY_SETUPS["realman"]["mic_pos"],
        dtype=torch.float32,
    )

    return mic_batch, gt_batch


def build_real_loader(
    batch_size,
    environment="OfficeRoom3",
    realman_root=None,
    snr_list=None,
    seed=42,
):
    from datasets.RealRecord import RealData

    if realman_root is None:
        realman_root = os.environ.get("REALMAN_ROOT", os.path.join("data", "RealMAN"))

    if snr_list is None:
        snr_list = [-10, -5, 0, 5, 10, 15]

    loaders = {}
    target_csv = os.path.join(realman_root, "test", "test_moving_source_location.csv")
    noise_dir = os.path.join(realman_root, "test", "ma_noise", environment)
    print(noise_dir)

    for snr_i in snr_list:
        realdata_test = RealData(
            data_dir=realman_root,
            target_dir=[target_csv],
            noise_dir=noise_dir,
            snr=[snr_i, snr_i],
            environment=environment,
            seed=seed,
        )

        loaders[snr_i] = DataLoader(
            realdata_test,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    return loaders

def save_file(mic_signal, acoustic_scene, sig_path, acous_path):
    
    if sig_path is not None:
        soundfile.write(sig_path, mic_signal, acoustic_scene['fs'])

    if acous_path is not None:
        file = open(acous_path,'wb')
        file.write(pickle.dumps(acoustic_scene))
        file.close()

def load_file(sig_path, acous_path):

    if sig_path is not None:
        mic_signal, fs = soundfile.read(sig_path)

    if acous_path is not None:
       with open(acous_path, 'rb') as file:
            acoustic_scene = pickle.load(file)


    if (sig_path is not None) & (acous_path is not None):
        return mic_signal, acoustic_scene
    elif (sig_path is not None) & (acous_path is None):
        return mic_signal
    elif (sig_path is None) & (acous_path is not None):
        return acoustic_scene

def cart2sph(cart, include_r=False):
    """ Cartesian coordinates to spherical coordinates conversion.
    Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
    where the radius is optional according to the include_r argument.
    """
    r = torch.sqrt(torch.sum(torch.pow(cart, 2), dim=-1))
    theta = torch.acos(cart[..., 2] / r)
    phi = torch.atan2(cart[..., 1], cart[..., 0])
    if include_r:
        sph = torch.stack((theta, phi, r), dim=-1)
    else:
        sph = torch.stack((theta, phi), dim=-1)
    return sph


def cart2sph_np(cart, include_r=True):
    xy2 = cart[..., 0]**2 + cart[..., 1]**2
    sph = np.zeros_like(cart)
    sph[..., 0] = np.sqrt(xy2 + cart[..., 2]**2)
    sph[..., 1] = np.arctan2(np.sqrt(xy2), cart[..., 2]) # Elevation angle defined from Z-axis down
    sph[..., 2] = np.arctan2(cart[..., 1], cart[..., 0])
    
    if include_r:
        return sph
    else:
        return sph[..., 1:]

def sph2cart(sph):
    """ Spherical coordinates to cartesian coordinates conversion.
    Each row contains one point in format (elevation, azimuth, radius),
    where the radius is supposed to be 1 if it is not included.
    """
    if sph.shape[-1] == 2: sph = torch.cat((sph, torch.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
    x = sph[..., 2] * torch.sin(sph[..., 0]) * torch.cos(sph[..., 1])
    y = sph[..., 2] * torch.sin(sph[..., 0]) * torch.sin(sph[..., 1])
    z = sph[..., 2] * torch.cos(sph[..., 0])
    return torch.stack((x, y, z), dim=-1)


def acoustic_power(s):
    """ Acoustic power of after removing the silences.
    """
    w = 512  # Window size for silent detection
    o = 256  # Window step for silent detection

    # Window the input signal
    s = np.ascontiguousarray(s)
    sh = (s.size - w + 1, w)
    st = s.strides * 2
    S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]

    window_power = np.mean(S ** 2, axis=-1)
    th = 0.01 * window_power.max()  # Threshold for silent detection
    return np.mean(window_power[np.nonzero(window_power > th)])

def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps, base_lr, eta_min=5e-5):
    """
    optimizer: torch optimizer (can have multiple param_groups)
    total_steps: nb_epoch * len(train_loader)
    warmup_steps: int, e.g., 0.05~0.1 of total_steps
    base_lr: your target peak lr (e.g., 5e-4)
    eta_min: min lr for cosine phase
    """
    # 1) Set all param_group lr to base_lr (scheduler will scale during warmup)
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr

    # 2) Warmup: linear ramp factor from 0 -> 1
    def warmup_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # 3) Cosine phase: after warmup, cosine from base_lr -> eta_min
    cosine_steps = max(1, total_steps - warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=eta_min
    )

    # 4) Use SequentialLR to switch schedulers at warmup_steps
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps]
    )
    return scheduler



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="min"):
        """
        Args:
            patience: number of epochs with no improvement before stopping
            min_delta: minimum change to qualify as an improvement
            mode: "min" for loss/MAE, "max" for accuracy-like metrics
        """
        assert mode in ["min", "max"]
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        self.best_model_path = None

    def _is_improved(self, score):
        if self.best_score is None:
            return True

        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)

    def step(self, score, epoch, trainer, run_dir, save_prefix="best"):
        """
        Args:
            score: current monitored metric
            epoch: current epoch index
            trainer: trainer object with save_checkpoint()
            run_dir: directory to save best checkpoint
            save_prefix: checkpoint filename prefix

        Returns:
            improved (bool): whether current score is a new best
        """
        improved = self._is_improved(score)

        if improved:
            old_best_model_path = self.best_model_path

            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

            self.best_model_path = os.path.join(run_dir, f"{save_prefix}_ep{epoch}.bin")
            trainer.save_checkpoint(self.best_model_path)

            if old_best_model_path is not None and os.path.exists(old_best_model_path):
                os.remove(old_best_model_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return improved

class Parameter:
    """ Random parammeter class.
    You can indicate a constant value or a random range in its constructor and then
    get a value acording to that with get_value(). It works with both scalars and vectors.
    """
    def __init__(self, *args):
        if len(args) == 1:
            self.random = False
            self.value = np.array(args[0])
            self.min_value = None
            self.max_value = None
        elif len(args) == 2:
            self. random = True
            self.min_value = np.array(args[0])
            self.max_value = np.array(args[1])
            self.value = None
        else: 
            raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
    
    def get_value(self):
        if self.random:
            return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
        else:
            return self.value
def generate_regular_polygon(n_sides, radius=1):
    """Generate a regular polygon with n_sides sides and radius radius."""

    points = []
    for i in range(n_sides):
        x = radius * math.cos(2 * math.pi * i / n_sides)
        y = radius * math.sin(2 * math.pi * i / n_sides)
        points.append([x, y])

    return torch.Tensor(points)
def forgetting_norm(input, num_frame_set=None):
    """
        Function: Using the mean value of the near frames to normalization
        Args:
            input: feature [B, C, F, T]
            num_frame_set: length of the training time frames, used for calculating smooth factor
        Returns:
            normed feature
        Ref: Online Monaural Speech Enhancement using Delayed Subband LSTM, INTERSPEECH, 2020
    """
    assert input.ndim == 4
    batch_size, num_channels, num_freqs, num_frames = input.size()
    input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

    if num_frame_set == None:
        num_frame_set = deepcopy(num_frames)

    mu = 0
    mu_list = []
    for frame_idx in range(num_frames):
        if frame_idx<=num_frame_set:
            alpha = (frame_idx - 1) / (frame_idx + 1)
        else:
            alpha = (num_frame_set - 1) / (num_frame_set + 1)
        current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(batch_size, 1) # [B, 1]
        mu = alpha * mu + (1 - alpha) * current_frame_mu
        mu_list.append(mu)
    mu = torch.stack(mu_list, dim=-1) # [B, 1, T]
    output = mu.reshape(batch_size, 1, 1, num_frames)

    return output
