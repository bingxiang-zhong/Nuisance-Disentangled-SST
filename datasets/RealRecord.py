
import random

import torch
import os

import webrtcvad
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from datasets.datautils import search_files, audiowu_high_array_geometry
from pathlib import Path
import random
from torch import Tensor
from pathlib import PurePosixPath




class RealData(Dataset):
    def __init__(self, data_dir, target_dir, environment, noise_dir=None, input_fs=48000,
                 use_mic_id=[1,  3,  5,  7,  0],target_fs=16000, snr=[-5, 15], win_size=1600, hop_rate=1,seed = 42):

        
        self.seed = seed
        self.ends = 'CH1.flac'
        self.data_paths = []
        self.all_targets = pd.DataFrame()
        self.env = environment
        self.wav_use_len = 4
        self.target_len = self.wav_use_len * 10
        for dir in target_dir:
            target = pd.read_csv(dir)
            self.data_paths += [data_dir + i for i in target['filename'].to_list()]
            self.all_targets = pd.concat([self.all_targets, target], ignore_index=True)

        self.all_targets.set_index('filename', inplace=True)

        if isinstance(self.env, list):
            # For list of environments - check if any environment is in the path
            self.data_paths = list(filter(
                lambda path: any(env in path for env in self.env),
                self.data_paths
            ))
            # For DataFrame filtering with multiple environments
            self.all_targets = self.all_targets[
                self.all_targets.index.str.contains('|'.join(self.env), case=False, regex=True)
            ]
        else:
            #  Single environment
            self.data_paths = list(filter(lambda path: self.env in path, self.data_paths))
            self.all_targets = self.all_targets[
                self.all_targets.index.str.contains(self.env, case=False, regex=False)
            ]

        self.noise_dir = noise_dir

        if noise_dir:
            self.noise_paths = search_files(noise_dir, flag=self.ends)

        self.target_fs = target_fs
        self.input_fs = input_fs
        self.SNR = snr
        self.pos_mics = audiowu_high_array_geometry()
        self.use_mic_id = use_mic_id
       

    def __len__(self):
        return len(self.data_paths)

    def cal_vad(self, sig, fs=16000, th=-2.5):
        window_size = int(0.1 * fs)
        num_windows = len(sig) // window_size
        energies = []
        times = []
        for i in range(num_windows):
            window = sig[i * window_size:(i + 1) * window_size]
            fft_result = np.fft.fft(window)
            fft_result = fft_result[:window_size // 2]
            freqs = np.fft.fftfreq(window_size, 1 / fs)[:window_size // 2]
            energy = np.sum(np.abs(fft_result[(freqs >= 0) & (freqs <= 8000)]) ** 2)
            energies.append(np.log10(energy + 1e-10))
        energies = np.array(energies)
        energies = np.where(energies < th, 0, 1)

        return torch.from_numpy(energies[:, np.newaxis])

    def select_mic_array_no_circle(self, pos_mics, rng):
        mic_id_list = np.arange(28)
        specific_group_1 = {0, 2, 4, 6, 24}
        specific_group_2 = {1, 3, 5, 7, 24}
        not_use_five_linear_mics = True
        while not_use_five_linear_mics:
            num_values_to_select = rng.integers(low=2, high=9)
            CH_list = list(rng.choice(mic_id_list, num_values_to_select, replace=False))
            mic_gemo = pos_mics[CH_list, :]
            # 2 types 5-mic circle array
            if set(CH_list) == specific_group_1 or set(CH_list) == specific_group_2:
                not_use_five_linear_mics = True
            else:
                not_use_five_linear_mics = False
        return CH_list, mic_gemo

    def seg_signal(self, signal, fs, rng, dp_signal, len_signal_s=4):
        signal_start = rng.integers(low=0, high=signal.shape[0] - (len_signal_s * fs))
        # print(signal_start,signal_start*fs//frame_size,(signal_start+len_signal_s*frame_size)*fs//frame_size)
        seg_signal = signal[signal_start:signal_start + (len_signal_s * fs), :]

        seg_dp_signal = dp_signal[signal_start:signal_start + (len_signal_s * fs)]
        return seg_signal, signal_start, seg_dp_signal

    def load_signals(self, sig_path, use_mic_id):

        channels = []
        for i in use_mic_id:
            temp_path = sig_path.replace('.flac', f'_CH{i}.flac')
            single_ch_signal, fs = sf.read(temp_path)
            channels.append(single_ch_signal)
        mul_ch_signals = np.stack(channels, axis=-1)

        return mul_ch_signals, fs

    def load_noise(self, noise_path, begin_index, end_index, use_mic_id):
        channels = []

        for i in use_mic_id:
            temp_path = noise_path.replace('_CH1.flac', f'_CH{i}.flac')
            try:
                single_ch_signal, fs = sf.read(temp_path, start=begin_index, stop=end_index)
            except:
                print(temp_path, begin_index, end_index)
            channels.append(single_ch_signal)
        mul_ch_signals = np.stack(channels, axis=-1)
        return mul_ch_signals, fs

    def resample(self, mic_signal, fs, new_fs):
        signal_resampled = signal.resample(mic_signal, int(mic_signal.shape[0] * new_fs / fs))
        return signal_resampled

    def get_snr_coff(self, wav1, wav2, target_dB):
        ae1 = np.sum(wav1 ** 2) / np.prod(wav1.shape)
        ae2 = np.sum(wav2 ** 2) / np.prod(wav2.shape)
        if ae1 == 0 or ae2 == 0 or not np.isfinite(ae1) or not np.isfinite(ae2):
            return None
        coeff = np.sqrt(ae1 / ae2 * np.power(10, -target_dB / 10))
        return coeff


    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed)

        sig_path = self.data_paths[idx]
        # print(self.data_paths,len(self.data_paths))

        use_mic_id_item = self.use_mic_id
        # cal vad
        dp_sig_path = sig_path.replace('ma_noisy_speech', 'dp_speech')
        dp_signal, dp_fs = sf.read(dp_sig_path)
        if dp_fs != self.target_fs:
            dp_signal = self.resample(mic_signal=dp_signal, fs=dp_fs, new_fs=self.target_fs)
        # print(dp_signal.shape)
        # sf.write('./dp_sig/' + str(idx)+'.wav',dp_signal,samplerate=self.target_fs)
        # load_path = sig_path
        load_path = sig_path.replace('ma_noisy_speech', 'ma_speech')
        mic_signal, fs = self.load_signals(load_path, use_mic_id=use_mic_id_item)
        if fs != self.target_fs:
            mic_signal = self.resample(mic_signal=mic_signal, fs=fs, new_fs=self.target_fs)
        len_signal = mic_signal.shape[0] / self.target_fs

        # pading or cut the source signal
        if len_signal < 5:
            input_length = int(self.wav_use_len * self.target_fs)
            input_mic_signal = np.zeros((input_length, mic_signal.shape[1]))
            min_length = min(input_length, mic_signal.shape[0])
            input_mic_signal[:min_length, :] = mic_signal[:min_length, :]
            dp_vad_temp = self.cal_vad(dp_signal)
            if dp_vad_temp.shape[0] > 40:
                dp_vad_temp = dp_vad_temp[:40, :]
            target = self.all_targets.at[sig_path.split('RealMAN/')[-1], 'angle(°)']
            if isinstance(target, float):
                targets = torch.ones((self.target_len, 1)) * int(target)
                vad_source = torch.zeros((self.target_len, 1))
                dp_vad = torch.zeros((self.target_len, 1))
                end_index = min(int(len_signal * 10), self.target_len)
                vad_source[:end_index] = 1
                dp_vad[:dp_vad_temp.shape[0], :] = dp_vad_temp
            elif isinstance(target, str):
                temp_targets = np.array([int(float(i)) for i in target.split(',')])
                targets = torch.zeros((self.target_len, 1))

                length_to_copy = min(len(temp_targets), self.target_len)
                targets[:length_to_copy, :] = torch.from_numpy(temp_targets[:length_to_copy, np.newaxis])
                vad_source = torch.zeros((self.target_len, 1))
                dp_vad = torch.zeros((self.target_len, 1))
                vad_source[:length_to_copy] = 1
                dp_vad[:dp_vad_temp.shape[0], :] = dp_vad_temp
            else:
                print(type(target))
                print(sig_path, target)
        else:
            input_mic_signal, signal_start, input_dp_signal = self.seg_signal(signal=mic_signal, fs=self.target_fs,
                                                                              dp_signal=dp_signal, rng=rng, len_signal_s=self.wav_use_len)
            dp_vad = self.cal_vad(input_dp_signal)
            target = self.all_targets.at[sig_path.split('RealMAN/')[-1], 'angle(°)']
            if isinstance(target, float):
                targets = torch.ones((self.target_len, 1)) * int(target)
                vad_source = torch.ones((self.target_len, 1))
            elif isinstance(target, str):
                targets = np.array([int(float(i)) for i in target.split(',')])
                targets_idx_begin = int(signal_start / (self.target_fs / 10))
                targets = torch.from_numpy(
                    targets[targets_idx_begin:targets_idx_begin + self.target_len, np.newaxis])
                vad_source = torch.ones((self.target_len, 1))
            else:
                print(sig_path, target)
   
        
        if self.noise_dir:
            snr_item = rng.uniform(self.SNR[0], self.SNR[1])
            
            noise_path = self.noise_paths[rng.integers(low=0, high=len(self.noise_paths))]
            wav_info = sf.info(noise_path)
            wav_frames = wav_info.frames
           
            noise_begin_index =  rng.integers(low=0, high=wav_frames-(self.wav_use_len*self.input_fs))
            noise_end_index =  noise_begin_index + (self.wav_use_len*self.input_fs)
            noise_signal,noise_fs = self.load_noise(noise_path,begin_index=noise_begin_index,end_index=noise_end_index,use_mic_id=use_mic_id_item)
           
            if noise_fs != self.target_fs:
                noise_signal = self.resample(noise_signal, noise_fs,self.target_fs)
            
            coeff =  self.get_snr_coff(input_mic_signal,noise_signal, snr_item)
            if not coeff:
                print("error")
            try:
                assert coeff is not None
            except:
                coeff = 1.0
            noise_signal = coeff * noise_signal
            input_mic_signal += noise_signal

        input_mic_signal -= input_mic_signal.mean()

        array_topo = self.pos_mics[use_mic_id_item]

        acoustic_scene = {

            "mic_pos": array_topo,
            "DOAw": targets.to(torch.float32),
            "vad": dp_vad.to(torch.float32),
            "fs":self.target_fs

        }
  
        return input_mic_signal, acoustic_scene


def collate_fn_gen(batch):
    mics, metas = zip(*batch)

    return np.stack(mics), list(metas)


def collate_fn_real( batch):
    """Collate function for the get_batch method.

    Args:
        mic_sig_batch (list): list of microphone signals (numpy arrays of shape (n_samples, n_mics)
                                                            or (n_frames, n_freq_bins, n_mics))
        acoustic_scene_batch (list): list of acoustic scenes

    Returns:
    """
    mics, metas = zip(*batch)


        # ---------- Mic signals ----------
    mic_batch = torch.as_tensor(
        np.stack(mics),
        dtype=torch.float32
    )  # (B, ...)

    # ---------- DOA ----------
    doa_batch = torch.as_tensor(
        np.stack([m["DOAw"] for m in metas]),
        dtype=torch.long
    )  # (B, nt, max_sources)
    # snr_batch  = torch.as_tensor(
    #     np.stack([m["SNR"] for m in metas]),
    #     dtype=torch.float32
    # )  # (B, nt, max_sources)

    # ---------- VAD ----------
    if "vad" in metas[0]:
        vad_np = np.stack([m["vad"] for m in metas])  # (B, nt, max_sources)
        vad_np = (vad_np.mean(axis=2) > (2.0 / 3.0))  # (B, nt) bool
        vad_batch = torch.as_tensor(vad_np, dtype=torch.bool).unsqueeze(-1)
    else:
        # Always active
        vad_batch = torch.ones(doa_batch.shape[:2], dtype=torch.bool).unsqueeze(-1)

    gt_batch = {}
    gt_batch["doa"] = doa_batch
    gt_batch["vad"] = vad_batch
    # gt_batch["snr"] = snr_batch

    return mic_batch, gt_batch
