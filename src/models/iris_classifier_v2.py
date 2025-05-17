import torch

from src.datasets import IrisDataset
from src.layers.linear import Linear
from src.layers.relu import ReLU
from src.loss.loss import CrossEntropyLoss
from src.models.trainer import Trainer
from src.optimizer.sgd import SGD
from src.sequential import Sequential

if __name__ == "__main__":
    torch.manual_seed(0)
    train_loader, test_loader = IrisDataset().train_loader(), IrisDataset().test_loader()

    model = Sequential(
        [
            Linear(4, 10),
            ReLU(),
            Linear(10, 3),
        ]
    )

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.get_params(), lr=0.1)

    trainer = Trainer(model, optimizer, criterion, train_loader, test_loader, False)

    trainer.train(100, patience=20)
