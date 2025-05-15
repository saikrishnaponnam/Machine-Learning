from torch import nn


def calculate_accuracy(model: nn.Module, test_loader) -> float:
    """
    Calculate the accuracy of the model on the test set.
    """
    correct = 0
    total = 0

    for x, y in test_loader:
        logits = model(x)
        _, predicted = logits.max(dim=1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    accuracy = correct / total
    return accuracy
