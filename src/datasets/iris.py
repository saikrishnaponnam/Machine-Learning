import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.dataset import Dataset


class IrisDataset(Dataset):

    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def train_loader(self):
        train_dataset = TensorDataset(torch.Tensor(self.X_train), torch.Tensor(self.y_train).to(dtype=torch.int32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader

    def test_loader(self):
        test_dataset = TensorDataset(torch.Tensor(self.X_test), torch.Tensor(self.y_test).to(dtype=torch.int32))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader
