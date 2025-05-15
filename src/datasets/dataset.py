from abc import abstractmethod, ABC


class Dataset(ABC):

    @abstractmethod
    def train_loader(self):
        """
        Returns the training data loader.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def test_loader(self):
        """
        Returns the test data loader.
        """
        raise NotImplementedError("Subclasses should implement this method.")
