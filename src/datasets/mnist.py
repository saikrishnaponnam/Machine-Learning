from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.datasets.dataset import Dataset


class MNISTDataset(Dataset):

    def __init__(self, batch_size=32):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.batch_size = batch_size

    def train_loader(self):
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def test_loader(self):
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=self.transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader
