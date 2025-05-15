from src.utils import calculate_accuracy


class Trainer:

    def __init__(self, model, optimizer, loss_fn, train_loader, test_loader=None, early_stopping=False):
        """
        Initialize the Trainer.

        Args:
            model: The model to be trained.
            optimizer: The optimizer to be used for training.
            loss_fn: The loss function to be used for training.
            train_loader: The DataLoader for the training data.
            test_loader: The DataLoader for the validation data (optional).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.early_stopping = early_stopping

    def train(self, epochs=10, patience=5):
        """
        Train the model.
        Args:
            epochs: The number of epochs to train for.
            patience: The number of epochs to wait for improvement before stopping. required if early_stopping is True.
        """

        if self.early_stopping:
            assert self.test_loader is not None, "Test loader is required for early stopping."
            best_acc = float("-inf")
            patience_counter = 0

        print("Training Size: ", len(self.train_loader.dataset))
        print("Training started...")

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.loss_fn(output, target)

                # Backward pass
                d_out = self.loss_fn.backward()
                self.model.backward(d_out)

                # Update weights
                self.optimizer.step()

                # if batch_idx % 10 == 0:
                #     print(f"Train Step: {batch_idx}, Loss: {loss.item()}")

            # Calculate accuracy
            acc = calculate_accuracy(self.model, self.test_loader)
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

            if self.early_stopping:
                if acc > best_acc:
                    patience_counter = 0
                    best_acc = acc
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1} with best acc {best_acc:.4f}")
                    break
