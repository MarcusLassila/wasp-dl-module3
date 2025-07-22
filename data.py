import torch
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

class CIFAR10(Dataset):
    
    def __init__(self, train=True):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0), # Scale data to values in [-1,1]
        ])
        self.train = train
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=self.transform)
    
    def __getitem__(self, index):
        item, _ = self.dataset[index] # Discard labels
        return item
    
    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    cifar10 = CIFAR10()
    print(len(cifar10))
    x = next(iter(cifar10))
    print(x.shape)
    print(x.min(), x.max())
    print(cifar10.train)