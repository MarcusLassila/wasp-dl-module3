import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import Dataset
import os, requests

class CIFAR10_1(Dataset):
    """
    CIFAR-10.1 (v4 or v6) from the official .npy files.
    Returns (img, label) where img is float tensor in [0,1], shape (3,32,32).
    """
    CLASSES = ("airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck")
    C10_1_URLS = {
        "v4": {
            "data":   "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v4_data.npy",
            "labels": "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v4_labels.npy",
        },
        "v6": {
            "data":   "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy",
            "labels": "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy",
        },
    }

    def __init__(self, root="./data/cifar10_1", version="v6", transform=None, return_labels=False):
        version = version.lower()
        assert version in {"v4","v6"}
        root = os.path.expanduser(root)
        self.transform = transform
        self.return_labels = return_labels

        data_path   = os.path.join(root, f"cifar10.1_{version}_data.npy")
        labels_path = os.path.join(root, f"cifar10.1_{version}_labels.npy")

        self._download(CIFAR10_1.C10_1_URLS[version]["data"], data_path)
        self._download(CIFAR10_1.C10_1_URLS[version]["labels"], labels_path)

        imgs = np.load(data_path)     # (N,32,32,3) uint8
        labels = np.load(labels_path) # (N,)
        self.images = torch.from_numpy(imgs).permute(0,3,1,2).to(torch.float64) / 255.0
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def _download(self, url, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.return_labels:
            return x, self.labels[idx]
        return x

class CIFAR10(Dataset):
    '''Test set of CIFAR10'''

    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        self.dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item

    def __len__(self):
        return len(self.dataset)
