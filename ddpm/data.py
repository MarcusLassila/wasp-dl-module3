import numpy as np
import torch
import tensorflow_datasets as tfds
from datasets import load_dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

class CIFAR10(Dataset):

    def __init__(self):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0), # Scale data to values in [-1,1]
        ])
        self.dataset = ConcatDataset([
            datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform),
            datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform),
        ])

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

class CelebAHQ(Dataset):

    def __init__(self):
        self.dataset = ConcatDataset([
            load_dataset("korexyz/celeba-hq-256x256", split="train"),
            load_dataset("korexyz/celeba-hq-256x256", split="validation"),
        ])
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),
        ])

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.dataset)

class Flowers(Dataset):

    def __init__(self):
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize((32, 32),
                     interpolation=T.InterpolationMode.BICUBIC,
                     antialias=True),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        self.dataset = load_dataset("huggan/flowers-102-categories")["train"]

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    dataset = CIFAR10_1()
    print(len(dataset))
    import utils
    img = dataset[1]
    print(img.min(), img.max(), img.shape)
    utils.plot_image(img, rescale_method="none")
