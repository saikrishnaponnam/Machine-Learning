import torch

from src.datasets import MNISTDataset
from src.layers.conv import Conv
from src.layers.flatten import Flatten
from src.layers.linear import Linear
from src.layers.relu import ReLU
from src.loss.loss import CrossEntropyLoss
from src.models.trainer import Trainer
from src.optimizer.sgd import SGD
from src.sequential import Sequential

if __name__ == "__main__":

    torch.manual_seed(0)
    train_loader, test_loader = MNISTDataset().train_loader(), MNISTDataset().test_loader()

    model = Sequential(
        [
            Conv(1, 8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Flatten(),
            Linear(28 * 28 * 8, 10),
        ]
    )

    model.summary()

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.get_params(), lr=0.01)

    trainer = Trainer(model, optimizer, criterion, train_loader, test_loader, False)
    trainer.train(10, patience=20)
