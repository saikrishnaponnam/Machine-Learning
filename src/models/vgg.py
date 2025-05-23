import torch

from src.datasets.cifar10 import CIFAR10Dataset
from src.layers.conv import Conv
from src.layers.flatten import Flatten
from src.layers.linear import Linear
from src.layers.max_pool import MaxPool
from src.layers.relu import ReLU
from src.loss.loss import CrossEntropyLoss
from src.models.trainer import Trainer
from src.optimizer.sgd import SGD
from src.sequential import Sequential

cfgs = {
    "vgg_11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg_13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg_16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg_19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def build_conv_layer(cfg):
    layers = []
    in_channels = 3
    for c in cfg:
        if c == "M":
            layers += [MaxPool(2, stride=2)]
        else:
            layers += [Conv(in_channels, c, kernel_size=3, stride=1, padding=1)]
            layers += [ReLU()]
            in_channels = c

    return layers


if __name__ == "__main__":

    torch.manual_seed(0)
    dataset = CIFAR10Dataset()
    train_loader, test_loader = dataset.train_loader(), dataset.test_loader()

    conv_layers = build_conv_layer(cfgs["vgg_13"])
    num_classes = 10

    model = Sequential(
        [
            *conv_layers,
            Flatten(),
            Linear(512 * 1 * 1, 4096),
            ReLU(),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, num_classes),
        ]
    )

    model.summary()

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.get_params(), lr=0.01)

    trainer = Trainer(model, optimizer, criterion, train_loader, test_loader, False)
    trainer.train(10, patience=20)
