import torch
from datasets import load_dataset
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

class CelebAHQ(Dataset):

    def __init__(self, train=True):
        self.split = "train" if train else "validation"
        self.dataset = load_dataset("korexyz/celeba-hq-256x256", split=self.split)
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

if __name__ == "__main__":
    celebahq = CelebAHQ(train=True)
    import utils
    img = celebahq[0]
    utils.plot_image(img, rescale_method="none")
