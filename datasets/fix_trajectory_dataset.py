import os
import pickle

import soundfile
from torch.utils.data import Dataset
from utils import load_file

class FixTrajectoryDataset(Dataset):
    def __init__(self, data_dir, dataset_sz, transforms=None):
        self.transforms = transforms
        self.data_paths = []
        data_names = os.listdir(data_dir)
        for fname in data_names:
            front, ext = os.path.splitext(fname)
            if ext == '.wav':
                full_path = os.path.join(data_dir, fname)
                self.data_paths.append(full_path)
                
        self.dataset_sz = len(self.data_paths) if dataset_sz is None else dataset_sz


    def __len__(self):
        return self.dataset_sz 
    def __getitem__(self, idx):
        if idx < 0: idx = len(self) + idx
        sig_path = self.data_paths[idx]
        acous_path = sig_path.replace('wav','npz')
        mic_signals, acoustic_scene = load_file(sig_path, acous_path)
   
        return mic_signals, acoustic_scene
