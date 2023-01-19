###################################################################################################
# American Sign Language (ASL) dataloader
# Aurore De Spirlet
# 2022 - ETH Zurich
###################################################################################################
"""
ASL dataset
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.io import read_image

import ai8x

import os
import pandas as pd

import matplotlib.pyplot as plt


"""
Custom image dataset class
"""
class AslDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_labels = pd.read_csv(os.path.join(img_dir, "labels.txt"))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, "all", self.img_labels.iloc[idx, 0])
        image = read_image(img_path)   #converts to tensor
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label



def asl_get_datasets(data, load_train=False, load_test=False):
   
    (data_dir, args) = data
    # data_dir = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),  #
            #transforms.RandomAffine(degrees=30, translate=(0.5, 0.5), scale=(0.5,1.5), fill=0),
            
            ############################
            # TODO: Add more transform #
            ############################
            
            #transforms.Resize((64,64)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = AslDataset(img_dir=os.path.join(data_dir, "asl", "train"), transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            # 960 and 720 are not random, but dimension of input test img
            #transforms.CenterCrop((960,720)),
            #transforms.Resize((64,64)),
            #transforms.RandomAffine(degrees=30, translate=(0.5, 0.5), scale=(0.5,1.5), fill=0),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            # transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = AslDataset(img_dir=os.path.join(data_dir, "asl", "test"), transform=test_transform)

        # if args.truncate_testset:
        #     test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


"""
Dataset description
"""
datasets = [
    {
        'name': 'asl',
        'input': (1, 28, 28),
        'output': list(map(str, range(25))),
        'loader': asl_get_datasets,
    }
]
