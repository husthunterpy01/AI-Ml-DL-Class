import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class CIFARDataset(Dataset):
    def __init__(self, root=".", train=True, transform=None):
        data_path = os.path.join(root, "cifar-10-batches-py")
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]
        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item].reshape((3, 32, 32)).astype(np.float32)
        if self.transform:
            image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
        label = self.labels[item]
        return image, label


if __name__ == '__main__':
    dataset = CIFARDataset(root="./data")
    index = 400
    image, label = dataset.__getitem__(index)
    print(image.shape)
    print(label)