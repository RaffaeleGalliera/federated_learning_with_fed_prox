from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from pathlib import Path
import torch
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        # Get images from from directory going recursively into directory/label/*.jpg
        self.images = [os.path.join(dp, f)
                       for dp, dn, fn in os.walk(os.path.expanduser(directory))
                       for f in fn]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')
        label = self._get_label(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_label(self, image_path):
        return torch.tensor(int(Path(image_path).parts[-2]))


class CustomDatasetWithLabelsList(CustomDataset):
    def __init__(self, directory, labels_list, transform=None):
        super().__init__(directory, transform)
        self.labels_list = labels_list
        self.images = [os.path.join(dp, f)
                       for dp, dn, fn in os.walk(os.path.expanduser(directory))
                       for f in fn
                       if Path(dp).parts[-1] in self.labels_list]


class CustomDatasetFromPOST(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_file = self.file_list[idx]
        image = Image.open(image_file.stream).convert('L')  # Convert to grayscale
        label = self._get_label(image_file)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_label(self, image_file):
        # Implement label extraction from image file if needed
        return 0  # Placeholder
