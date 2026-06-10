import math
import os
import random


import numpy as np
import scipy
import soundfile
import torch

class NoiseDataset():
    def __init__(self, T, fs, nmic, noise_type, noise_path=None, c=343.0):
        self.T = T
        self.fs = fs
        self.nmic = nmic
        self.noise_type = noise_type  # ? 'diffuse' and 'real_world' cannot exist at the same time
        # self.mic_pos = mic_pos # valid for 'diffuse'
        self.noie_path = noise_path  # valid for 'diffuse' and 'real-world'
        if noise_path != None:
            _, self.path_set = self._exploreCorpus(noise_path, 'wav')
        self.c = c

    @staticmethod
    def _repeat_to_min_length(noise, nsample_desired):
        repeats = math.ceil(nsample_desired / noise.shape[0])
        if repeats == 1:
            return np.array(noise, copy=True)
        return np.concatenate([noise] * repeats, axis=0)

    def get_random_noise(self, mic_pos=None):
        # noise_type = self.noise_type.getValue()
        noise_type = self.noise_type

        if noise_type == 'spatial_white':
            noise_signal = self.gen_Gaussian_noise(self.T, self.fs, self.nmic)

        elif noise_type == 'diffuse':
            idx = random.randint(0, len(self.path_set) - 1)
            noise, fs = soundfile.read(self.path_set[idx])
            if fs != self.fs:
                # noise = librosa.resample(noise, orig_sr = fs, target_sr = self.fs)
                noise = scipy.signal.resample_poly(noise, up=self.fs, down=fs)

            nsample_desired = int(self.T * self.fs * self.nmic)
            noise_copy = self._repeat_to_min_length(noise, nsample_desired)
            nsample = noise_copy.shape[0]

            st = random.randint(0, nsample - nsample_desired)
            ed = st + nsample_desired
            noise_copy = noise_copy[st:ed]

            noise_signal = self.gen_diffuse_noise(noise_copy, self.T, self.fs, mic_pos, c=self.c)
        elif noise_type == 'real_world':  # the array topology should be consistent
            idx = random.randint(0, len(self.path_set) - 1)
            noise, fs = soundfile.read(self.path_set[idx])
            nmic = noise.shape[-1]
            if nmic != self.nmic:
                raise Exception('Unexpected number of microphone channels')
            if fs != self.fs:
                # noise = librosa.resample(noise.transpose(1,0), orig_sr = fs, target_sr = self.fs).transpose(1,0)
                noise = scipy.signal.resample_poly(noise, up=self.fs, down=fs)
            nsample_desired = int(self.T * self.fs)
            noise_copy = self._repeat_to_min_length(noise, nsample_desired)
            nsample = noise_copy.shape[0]

            st = random.randint(0, nsample - nsample_desired)
            ed = st + nsample_desired
            noise_signal = noise_copy[st:ed, :]

        else:
            raise Exception('Unknown noise type specified')

        return noise_signal

    def _exploreCorpus(self, path, file_extension):
        directory_tree = {}
        directory_path = []
        for item in os.listdir(path):
            if os.path.isdir(os.path.join(path, item)):
                directory_tree[item], directory_path = self._exploreCorpus(os.path.join(path, item), file_extension)
            elif item.split(".")[-1] == file_extension:
                directory_tree[item.split(".")[0]] = os.path.join(path, item)
                directory_path += [os.path.join(path, item)]
        return directory_tree, directory_path

    def gen_Gaussian_noise(self, T, fs, nmic):

        noise = np.random.standard_normal((int(T * fs), nmic))

        return noise

    def gen_diffuse_noise(self, noise, T, fs, mic_pos, nfft=256, c=343.0, type_nf='spherical', device='cuda'):
        """PyTorch implementation with full GPU acceleration and optimized FFT operations"""
        # Move inputs to GPU
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

        M = mic_pos.shape[0]
        L = int(T * fs)

        # Generate M mutually 'independent' input signals
        if isinstance(noise, np.ndarray):
            noise = torch.from_numpy(noise).float().to(device)

        noise = noise - torch.mean(noise)
        noise_M = noise[:M * L].reshape(M, L).transpose(0, 1)

        # Generate matrix with desired spatial coherence
        n_freq = nfft // 2 + 1
        ww = 2 * math.pi * self.fs * torch.arange(n_freq, device=device).float() / nfft
        DC = torch.zeros(M, M, n_freq, device=device)
        freq_ones = torch.ones(n_freq, device=device)

        if isinstance(mic_pos, np.ndarray):
            mic_pos = torch.from_numpy(mic_pos).float().to(device)

        # Compute distances once and reuse
        distances = torch.zeros(M, M, device=device)
        for p in range(M):
            for q in range(p, M):  # Use symmetry to compute only half the matrix
                if p == q:
                    distances[p, q] = 0.0
                else:
                    dist = torch.norm(mic_pos[p, :] - mic_pos[q, :])
                    distances[p, q] = dist
                    distances[q, p] = dist  # Matrix is symmetric

        # Vectorized computation for coherence matrix
        for p in range(M):
            for q in range(M):
                if p == q:
                    DC[p, q, :] = freq_ones
                else:
                    dist = distances[p, q]
                    if type_nf == 'spherical':
                        # PyTorch sinc implementation
                        x = ww * dist / (c * math.pi)
                        DC[p, q, :] = torch.where(x == 0, torch.tensor(1.0, device=device),
                                                  torch.sin(x) / x)
                    elif type_nf == 'cylindrical':
                        # Efficient Bessel function approximation
                        x = ww * dist / c

                        # Polynomial approximation for Bessel J0 (optimized computation)
                        x_sq = x * x
                        j0 = 1.0 - x_sq / 4.0 * (1.0 - x_sq / 16.0 * (1.0 - x_sq / 36.0 * (1.0 - x_sq / 64.0)))
                        DC[p, q, :] = j0
                    else:
                        raise Exception('Unknown noise field')

        # Generate sensor signals with desired spatial coherence
        noise_signal = self.mix_signals_efficient(noise_M, DC, nfft=nfft, device=device)

        return noise_signal.cpu().numpy()

    def mix_signals_efficient(self, noise, DC, method='cholesky', nfft=256, device='cuda'):
        """Optimized PyTorch implementation using torch.stft and torch.istft"""
        M = noise.shape[1]  # Number of sensors
        L = noise.shape[0]  # Signal length

        # Calculate number of frequency bins and STFT parameters
        n_fft = nfft  # Use the provided nfft as n_fft
        hop_length = n_fft // 4  # 75% overlap
        win_length = n_fft

        # Create Hann window
        window = torch.hann_window(win_length, device=device)

        # Transpose noise to [M, L] for torch.stft
        noise = noise.transpose(0, 1)

        # Use PyTorch's built-in STFT
        # Shape: [M, n_fft//2+1, n_frames, 2] where 2 is for real and imaginary parts
        stft_real_imag = torch.stft(
            noise,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True
        )

        # Shape becomes [M, n_fft//2+1, n_frames]
        n_frames = stft_real_imag.shape[2]

        # Prepare output tensor for modified STFT
        X = torch.zeros_like(stft_real_imag)
        eye = torch.eye(M, device=device)

        # Process each frequency bin
        for k in range(1, n_fft // 2 + 1):
            current_DC = DC[:, :, k]

            if method == 'cholesky':
                # Add small value to diagonal for numerical stability
                current_DC = current_DC + eye * 1e-6
                try:
                    # Convert to complex before decomposition
                    C = torch.linalg.cholesky(current_DC.to(torch.complex64))
                except:
                    # Fallback if still not positive definite
                    current_DC = current_DC + eye * 1e-4
                    C = torch.linalg.cholesky(current_DC.to(torch.complex64))
            elif method == 'eigen':
                D, V = torch.linalg.eigh(current_DC)
                # Ensure positive eigenvalues and sort
                D = torch.clamp(D, min=1e-10)
                idx = torch.argsort(D, descending=True)
                D = D[idx]
                V = V[:, idx]
                # Create diagonal matrix of eigenvalues
                D_sqrt = torch.diag(torch.sqrt(D))
                # Compute mixing matrix
                C = torch.matmul(V, D_sqrt).to(torch.complex64)
            else:
                raise Exception('Unknown method specified')

            # Process each frame for the current frequency bin
            frames_k = stft_real_imag[:, k, :]  # [M, n_frames]

            # Apply mixing matrix to transform the signals
            # We need to handle the batch dimension (n_frames)
            # Reshape frames_k to [M, n_frames]
            # Apply mixing and reshape back
            X[:, k, :] = torch.matmul(C.conj(), frames_k)


        # Use PyTorch's built-in ISTFT
        output = torch.istft(
            X,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=L
        )

        # Transpose back to [L, M]
        return output.transpose(0, 1)

