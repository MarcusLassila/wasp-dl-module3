import torch
from datasets import load_dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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
            T.Resize((64, 64),
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

def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0

if __name__ == "__main__":
    flowers = Flowers()
    print(len(flowers))
    import utils
    img = flowers[0]
    print(img.min(), img.max(), img.shape)
    utils.plot_image(img, rescale_method="none")
