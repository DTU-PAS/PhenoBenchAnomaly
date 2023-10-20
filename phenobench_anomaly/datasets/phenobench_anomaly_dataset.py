import numpy as np
from phenobench.phenobench_loader import PhenoBench
import os
from PIL import Image
import torch


class PhenoBenchAnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, weed_percentage, transform=None):
        self.root = root_dir
        self.split = split
        self.transform = transform
        print(os.path.dirname(os.path.realpath(__file__)))
        if split == "train":
            extension_file = f"{os.path.dirname(os.path.realpath(__file__))}/../../PhenoBench_extensions/{split}/phenobench_anomaly_{weed_percentage}.txt"
        elif split == "val":
            extension_file = f"{os.path.dirname(os.path.realpath(__file__))}/../../PhenoBench_extensions/{split}/phenobench_anomaly.txt"
        
        with open(extension_file, "r") as f:
            self.datapoints = f.readlines()

    def __getitem__(self, idx):
        datapoint = self.datapoints[idx].split()

        image = np.array(Image.open(os.path.join(self.root, self.split, "images", datapoint[0])).convert("RGB"))
        image = image[int(datapoint[1]):int(datapoint[3]), int(datapoint[2]):int(datapoint[4])]

        semantics = np.array(Image.open(os.path.join(self.root, self.split, "semantics", datapoint[0])))
        semantics = semantics[int(datapoint[1]):int(datapoint[3]), int(datapoint[2]):int(datapoint[4])]
        # Ignoring information of partial plants
        semantics[semantics == 3] = 1
        semantics[semantics == 4] = 2

        # Transform data
        if self.transform:
            transformed = self.transform(image=image, mask=semantics)
            image, semantics = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2, 0, 1)
        
        sample = {}
        sample["image_name"] = datapoint[0]
        sample["tile"] = [int(datapoint[1]), int(datapoint[2]), int(datapoint[3]), int(datapoint[4])]
        sample["image"] = image
        sample["semantics"] = semantics
        return sample

    def __len__(self):
        return len(self.datapoints)
