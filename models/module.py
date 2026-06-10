import numpy as np
import torch
import torch.nn as nn


class STFT(nn.Module):
    """Get STFT coefficients of microphone signals."""

    def __init__(self, win_len, win_shift_ratio, nfft, win="hann"):
        super().__init__()
        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft
        self.win = win

    def forward(self, signal):
        nsample = signal.shape[-2]
        nch = signal.shape[-1]
        win_shift = int(self.win_len * self.win_shift_ratio)
        nf = int(self.nfft / 2) + 1

        nb = signal.shape[0]
        nt = np.floor(nsample / win_shift + 1).astype(int)
        stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64, device=signal.device)

        if self.win == "hann":
            window = torch.hann_window(window_length=self.win_len, device=signal.device)
        else:
            window = None

        for ch_idx in range(nch):
            stft[:, :, :, ch_idx] = torch.stft(
                signal[:, :, ch_idx],
                n_fft=self.nfft,
                hop_length=win_shift,
                win_length=self.win_len,
                window=window,
                center=True,
                normalized=False,
                return_complex=True,
            )

        return stft


def build_norm_layer(norm_type, num_channels, num_groups=8):
    norm_type = norm_type.lower()

    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "gn":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    raise ValueError(f"Unsupported norm type: {norm_type}")


class CausCnnBlock(nn.Module):
    """Basic convolutional block with configurable normalization."""

    def __init__(
        self,
        inplanes,
        planes,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        use_res=True,
        downsample=None,
        norm_type="gn",
        num_groups=8,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = build_norm_layer(norm_type, planes, num_groups)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm2 = build_norm_layer(norm_type, planes, num_groups)
        self.downsample = downsample
        self.use_res = use_res

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out = out + residual

        out = self.relu(out)
        return out
