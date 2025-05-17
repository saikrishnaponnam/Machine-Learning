import torch

from src.datasets import MNISTDataset
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
            Flatten(),
            Linear(28 * 28, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 10),
        ]
    )

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.get_params(), lr=0.01)

    trainer = Trainer(model, optimizer, criterion, train_loader, test_loader, False)

    trainer.train(10, patience=20)
