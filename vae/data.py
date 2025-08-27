import torch
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

if __name__ == "__main__":
    dataset = MNIST()
    sample = dataset[0]
    print(sample.shape)
    import matplotlib.pyplot as plt
    plt.imshow(sample.permute(1,2,0))
    plt.show()