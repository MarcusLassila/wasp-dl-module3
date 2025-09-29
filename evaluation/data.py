import numpy as np
import torch
import tensorflow_datasets as tfds
from torchvision import transforms as T
from torch.utils.data import Dataset

class CIFAR10_1(Dataset):

    def __init__(self, transform=None):
        # Load CIFAR-10.1 v6 (~2,000 images)
        ds = tfds.load("cifar10_1/v6", split="test", as_supervised=True)

        self.dataset = torch.from_numpy(np.array([x for x, _ in tfds.as_numpy(ds)])).permute(0, 3, 1, 2).float() / 255  # (N,3,32,32)
        self.transform = transform

    def __getitem__(self, index):
        item = self.dataset[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.dataset)
