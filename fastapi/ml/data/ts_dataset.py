import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, window_size=256):
        self.window_size = window_size
        self.data = []

        for seq in sequences:
            seq = self._normalize(seq)
            if len(seq) >= window_size:
                for i in range(0, len(seq) - window_size + 1, window_size):
                    self.data.append(seq[i:i + window_size])

    def _normalize(self, x):
        x = np.array(x, dtype=np.float32)
        return (x - x.mean()) / (x.std() + 1e-6)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return torch.tensor(x).unsqueeze(0)
        