"""
Torchvision dataloader utilities for 3D tensors
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
import warnings
import random
import scipy.ndimage as snd
import pydicom as dicom

# import torchio as tio
import traceback
import kornia.augmentation as ka
from kornia.augmentation import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    CenterCrop,
    RandomAffine,
    RandomGaussianBlur,
    RandomMotionBlur,
    RandomThinPlateSpline,
    RandomElasticTransform,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

warnings.filterwarnings("ignore")


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, mode):
        super().__init__()

        self.train_transforms = ka.AugmentationSequential(
            RandomHorizontalFlip(p=0.5),
            RandomThinPlateSpline(p=0.5),
            RandomVerticalFlip(p=0.5),
            CenterCrop((224, 224), p=1),
            RandomElasticTransform(),
            RandomAffine((-20, 20), p=0.5),
            RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
            RandomMotionBlur(3, 35.0, 0.5, p=0.5),
            data_keys=["input"],
            same_on_batch=False,
        )
        self.val_transforms = ka.AugmentationSequential(
            CenterCrop((224, 224), p=1),
            data_keys=["input"],
            same_on_batch=False,
        )
        self.mode = mode

    def forward(self, *kwargs):
        if self.mode.lower() == "train":
            x_out = self.train_transforms(*kwargs)  # BxCxHxW
        else:
            # x_out = kwargs
            x_out = self.val_transforms(*kwargs)
        return x_out


class Dataset2D(Dataset):

    """
    Multi-threaded 3D patch loader

    Attributes:
        root_dir (string): path to folder containing image patches and label patches
        image_list (list): populates a sorted list of images to be loaded
        transform (list): list of augmentations/transforms to be applied to the images
    """

    def __init__(self, image_list, augment, mode="train"):
        """
        Initializing 3D data loader
        Args:
            root_dir (string): path to folder containing image patches and label patches
            transform (None, (optional)list): list of augmentations/transforms to be applied to the images
        """
        self.image_list = image_list
        self.augment = augment
        self.mode = mode

    def __len__(self):
        """
        Computes length of dataset
        Returns:
            (int): length of `image_list`
        """
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Loads an image-label pair, augments and returns the sample
        Args:
            idx (int): index of the file from image_list to be loaded

        Returns:
            sample (dict): {'image': (numpy.ndarray),
            'label': (numpy.ndarray), 'ID': filename}
        """
        try:
            if ".npy" in self.image_list[idx]:
                image = np.load(self.image_list[idx])
                label = int(self.image_list[idx].replace(".npy", "").split("_x_")[-1])
            else:
                image = dicom.read_file(self.image_list[idx]).pixel_array
                label = 1
        except Exception as e:
            traceback.print_exc()
            image = np.zeros((512, 512))
            label = 0
            # raise e

        # image = np.clip(image, -100, 400)
        # image = (image + 100) / 500
        image = np.clip(image, 0, 20000)
        image = image / 20000.0
        # zoom_facs = np.array((512, 512)) / image.shape
        # image = snd.zoom(image, zoom_facs, order=1)
        if np.min(image.shape) < 224:
            diff = 224 - np.min(image.shape[0])
            image = np.pad(image, diff, mode="constant", constant_values=0)
        image = torch.from_numpy(image).float()
        # if self.mode == "train":
        image = self.augment(image)
        image, label = torch.squeeze(image)[None, :, :], torch.tensor(label)
        sample = {"image": image, "label": label, "ID": self.image_list[idx]}

        return sample


class RandomRotate(object):
    """Function for random rotation by right angles"""

    def __call__(self, sample):
        image = sample["image"]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        image = np.float32(image)
        label = np.float32(sample["label"])
        rot_list = [0, 1, 3, 0]
        angle_choice = random.choice(rot_list)
        image = np.rot90(image, k=angle_choice, axes=(1, 2))
        label = np.rot90(label, k=angle_choice, axes=(0, 1))
        sample["image"] = image
        sample["label"] = label

        return sample


class RandomScaleCrop(object):
    """Function for random scale and crop by right angles"""

    def __init__(self, scales, prob):
        self.scales = scales
        self.prob = 1.0 - prob

    def __call__(self, sample):

        image = sample["image"]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        label = sample["label"].astype(np.uint8)
        if random.random() > self.prob:
            size_original = image.shape[-1]

            scale = random.choice(self.scales)

            image = snd.zoom(image, [1.0, scale, scale], order=1)
            label = snd.zoom(label, [scale, scale], order=0)
            # edge = snd.zoom(edge, [scale, scale], order=0)
            size_zoomed = image.shape[-1]

            start_index = (size_zoomed // 2) - (size_original // 2)
            end_index = (size_zoomed // 2) + (size_original - size_original // 2)

            image = image[:, start_index:end_index, start_index:end_index]
            label = label[start_index:end_index, start_index:end_index]
            # edge = edge[start_index:end_index, start_index:end_index]

        sample["image"] = image
        sample["label"] = label
        return sample


class RandomFlip(object):
    """docstring for RandomFlip"""

    def __init__(self, prob):
        self.prob = 1 - prob

    def __call__(self, sample):

        image = sample["image"]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        image = np.float32(image)
        label = sample["label"].astype(np.uint8)
        if random.random() > self.prob:
            flip_axis = [0, 1, 2]
            axis_choice = random.choice(flip_axis)
            if axis_choice == 0:
                image = np.flip(image, 0)
            else:
                image = np.flip(image, axis_choice)
                label = np.flip(label, axis_choice - 1)
        sample["image"] = image
        sample["label"] = label

        return sample


class Appendcrops(object):
    """Function for Appending crops"""

    def __call__(self, sample):

        image = sample["image"]
        label = sample["label"]
        # edge = sample["edge"]

        image_zoomed = snd.zoom(image, [1.0, 2.0, 2.0], order=1)

        center_x, center_y = (
            image_zoomed.shape[1] // 2,
            image_zoomed.shape[2] // 2,
        )
        len_x, len_y = label.shape[0] // 2, label.shape[1] // 2

        crop1 = image_zoomed[9, :center_x, :center_y]
        crop2 = image_zoomed[9, center_x:, :center_y]
        crop3 = image_zoomed[9, center_x:, center_y:]
        crop4 = image_zoomed[9, :center_x, center_y:]
        crop5 = image_zoomed[
            9,
            center_x - len_x : center_x + len_x,
            center_y - len_y : center_y + len_y,
        ]

        image = np.concatenate(
            (
                image,
                np.expand_dims(crop1, axis=0),
                np.expand_dims(crop2, axis=0),
                np.expand_dims(crop3, axis=0),
                np.expand_dims(crop4, axis=0),
                np.expand_dims(crop5, axis=0),
            ),
            axis=0,
        )

        sample["image"] = image
        sample["label"] = label
        # sample["edge"] = edge

        return sample


class ToTensor(object):

    """
    Function that converts 3D numpy array to 4D Torch tensor of shape [1, num_slices, height, width]
    """

    def __call__(self, sample):
        """
        Args:
            sample (dict): format -  {'image': (3D numpy array),
                                    'label': (3D numpy array),
                                    'ID': filename}

        Returns:
            sample (dict): format -  {'image': (4D torch tensor),
                                    'label': (4D torch tensor),
                                    'ID': filename}
        """
        image = sample["image"]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        image = np.float32(np.squeeze(image))
        label = np.float32(np.squeeze(sample["label"]))

        return {
            "image": torch.from_numpy(image.copy()),
            "label": torch.from_numpy(label.copy()),
            "ID": sample["ID"],
        }


def provider(config, mode="train"):
    """
    data provider for training neural networks in PyTorch for 3D images

    Args:
        mode (str, optional): 'train' or 'val' mode

    Returns:
        dataloader (torchvision dataloader)
    """
    train_files = np.load(config["train_list_path"])
    val_files = np.load(config["val_list_path"])

    print("Train files - ", len(train_files))
    print("Val files -", len(val_files))

    # t_train = [
    #     tio.RandomMotion(p=0.5, include=["image", "label"]),
    #     tio.RandomFlip(axes=("LR",), flip_probability=0.4, include=["image", "label"]),
    #     tio.RandomBlur(p=0.5, include=["image"]),
    #     tio.RandomAffine(
    #         p=0.3, image_interpolation="bspline", include=["image", "label"]
    #     ),
    #     tio.RandomElasticDeformation(
    #         num_control_points=7, max_displacement=7.5, include=["image", "label"]
    #     ),
    #     ToTensor()
    # ]
    t_train = DataAugmentation(mode="train")

    t_valid = DataAugmentation(mode="valid")

    if mode.lower() == "train":
        # dataset = Dataset2D(train_files, tio.Compose(t_train))
        dataset = Dataset2D(train_files, t_train, mode)
        batch_size = config["train_batch_size"]
        num_workers = config["num_workers_train"]
    else:
        dataset = Dataset2D(val_files, t_valid, mode)
        batch_size = config["val_batch_size"]
        num_workers = config["num_workers_val"]

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    return dataloader


if __name__ == "__main__":

    import time

    config = {
        "train_list_path": "/DDNstorage/users/krish/head_clsfcn/train_files_PT.npy",
        "val_list_path": "/DDNstorage/users/krish/head_clsfcn/val_files_PT.npy",
        "train_batch_size": 40,
        "val_batch_size": 100,
        "num_workers_train": 160,
        "num_workers_val": 180,
    }

    dataloader = provider(config, mode="train")

    total_batches = len(dataloader)
    start_time = time.time()
    # aug = DataAugmentation(mode="train")
    s_time = time.time()
    for i, batch in enumerate(dataloader):
        image, label = batch["image"], batch["label"]
        # label = label[:, np.newaxis, :, :]
        # a_time = time.time()
        # image, label = aug(image, label)
        # b_time = time.time()
        # print(batch["image"].size(), batch["label"].size())
        print(
            "Batch - {:2}/{:2}, time per batch - ".format(i + 1, total_batches),
            round(time.time() - s_time, 3),
            "secs",
            # ", time for aug - ",
            # round(b_time-a_time, 2)
        )
        s_time = time.time()

        pass
    print("Total time taken - ", round(time.time() - start_time), "secs")
