from phenobench_anomaly.datasets.phenobench_anomaly_dataset import PhenoBenchAnomalyDataset
import random
from matplotlib import pyplot as plt
import hydra
import os
import numpy as np
import albumentations as A
@hydra.main(config_path="../config", config_name="config.yaml")
def main(cfg):
    split = "train"
    weed_percentage = 0.01
    n_samples = 4

    DS_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    DS_STD = np.array([58.395, 57.120, 57.375]) / 255

    image_transform = A.Compose([
        # A.OneOf([A.Resize(width=341, height=341),
        # A.RandomResizedCrop(height=341, width=341, scale=(0.2, 1.0), p=1.0),], p=1.0),
        # A.HorizontalFlip(p=1.0),
        # A.VerticalFlip(p=1.0),
        # A.Rotate(p=0.5),
        # A.RGBShift(p=1.0),
        # A.RandomBrightness(p=1.0),
        # A.RandomContrast(p=1.0),
        # A.OneOf([A.GaussianBlur(p=1.0),
        #          A.Blur(p=1.0),
        #          A.MedianBlur(p=1.0),
        #          A.MotionBlur(p=1.0),
        #          ], p=0.4),
        A.Normalize(mean=DS_MEAN, std=DS_STD),
    ])

    train_data = PhenoBenchAnomalyDataset(cfg.phenobench_root, split, weed_percentage, transform=image_transform)

    fig, ax = plt.subplots(nrows=n_samples, ncols=2, figsize=(5, 10))
    for i in range(n_samples):
        sample = train_data[random.randint(0, len(train_data))]
        unnormalized_image = (sample["image"] * np.array(DS_STD)[:, None, None]) + np.array(DS_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        ax[i, 0].imshow(unnormalized_image)
        ax[i, 1].imshow(sample["semantics"])
    plt.savefig(f"{os.path.dirname(os.path.realpath(__file__))}/../phenobench_anomaly_visualization.png")


if __name__ == "__main__":
    main()