import torch
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import ConcatDataset, DataLoader, Dataset

class MNIST(Dataset):
    
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        self.dataset = ConcatDataset([
            datasets.MNIST(root='../data', train=True, download=True, transform=self.transform),
            datasets.MNIST(root='../data', train=False, download=True, transform=self.transform),
        ])
    
    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item
    
    def __len__(self):
        return len(self.dataset)
