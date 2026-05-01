
from src.core.interfaces import AbstractDataLoader
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize


class MNISTDataLoader(AbstractDataLoader):
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        data_root: str = "./data"
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_root = data_root

    def load_data(self):
        print("📥 Downloading/Loading MNIST Data...")

        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        train_data = MNIST(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform
        )

        test_data = MNIST(
            root=self.data_root,
            train=False,
            download=True,
            transform=transform
        )
        print("done downloading")

        return (
            DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle),
            DataLoader(test_data, batch_size=self.batch_size)
        )
