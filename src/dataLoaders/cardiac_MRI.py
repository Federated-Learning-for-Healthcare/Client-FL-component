import os
from glob import glob
from torch.utils.data import DataLoader
from src.core.interfaces import AbstractDataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from src.core.interfaces import AbstractDataLoader
from PIL import Image
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

def partition_dataset(image_paths, labels, num_partitions):
    partitions = [[] for _ in range(num_partitions)]
    label_partitions = [[] for _ in range(num_partitions)]

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        partitions[i % num_partitions].append(path)
        label_partitions[i % num_partitions].append(label)

    return list(zip(partitions, label_partitions))



class CardiacMRIDataLoader(AbstractDataLoader):
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        data_root: str = "./data",
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_root = data_root

    def load_data(self):
        print("📥 Loading CAD Cardiac MRI Dataset ...")

        normal_root = os.path.join(self.data_root, "MRI", "Normal")
        sick_root = os.path.join(self.data_root, "MRI", "Sick")

        normal_images = glob(os.path.join(normal_root, "**", "*.jpg"), recursive=True)
        sick_images = glob(os.path.join(sick_root, "**", "*.jpg"), recursive=True)

        all_images = normal_images + sick_images
        labels = [0] * len(normal_images) + [1] * len(sick_images)

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            all_images,
            labels,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        num_partitions = 10  # Example: partition into 5 clients
        partition_id = 0  # Example: select the first partition for this client

        partitioned_data = partition_dataset(all_images, labels, num_partitions)

        image_paths_partition, labels_partition = partitioned_data[partition_id]

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths_partition, labels_partition, test_size=0.2, stratify=labels_partition, random_state=42
        )

        train_dataset = MRIDataset(train_paths, train_labels, transform)
        test_dataset = MRIDataset(test_paths, test_labels, transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return (train_loader, test_loader)
