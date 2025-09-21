import numpy as np
import torch
import tensorflow_datasets as tfds
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import ConcatDataset, DataLoader, Dataset

class MNIST(Dataset):

    def __init__(self, train=True):
        self.transform = T.Compose([
            T.Resize(32), # Resize to 32 for architectural convenience
            T.ToTensor(),
        ])
        self.train = train
        self.dataset = datasets.MNIST(root='./data', train=train, download=True, transform=self.transform)

    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item

    def __len__(self):
        return len(self.dataset)

class CIFAR10(Dataset):

    def __init__(self, train=True):
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        self.train = train
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=self.transform)

    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item

    def __len__(self):
        return len(self.dataset)

class CIFAR10_1(Dataset):

    def __init__(self, transform=None):
        # Load CIFAR-10.1 v6 (~2,000 images)
        ds = tfds.load("cifar10_1/v6", split="test", as_supervised=True)

        self.dataset = torch.from_numpy(np.array([x for x, _ in tfds.as_numpy(ds)])).permute(0, 3, 1, 2)  # (N,3,32,32)
        self.transform = transform

    def __getitem__(self, index):
        item = self.dataset[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    dataset = MNIST()
    sample = dataset[0]
    print(sample.shape)
    import matplotlib.pyplot as plt
    plt.imshow(sample.permute(1,2,0))
    plt.show()