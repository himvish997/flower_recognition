import torch
import torchvision
import numpy as np
import os
import pandas as pd
import cv2
import efficientnet_pytorch
from torch.utils.data import DataLoader, Dataset



class Arguments():
    def __init__(self):
        self.labels_csv = '/mnt/NewVolume/projects/flower_recognition/data/train.csv'
        self.images_dir = '/mnt/NewVolume/projects/flower_recognition/data/train'

        self.batch_size = 16
        self.IMG_SIZE = (300, 300)

        self.cuda = False
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")


def read_csv(labels_csv, images_dir):
    df = pd.read_csv(labels_csv)
    df['labels'] = df['category'].map(lambda x: int(x) - 1)
    df['images'] = df['image_id'].map(lambda x: images_dir + "/{}.jpg".format(x))
    df['exists'] = df['images'].map(os.path.exists)
    df = df[df['exists']]
    df = df.drop(['category', 'image_id', 'exists'], axis=1)
    return df


class MyDataset(Dataset):
    def __init__(self, dataframe, IMG_SIZE):
        self.labels = dataframe['labels']
        self.images = dataframe['images']

        self.IMG_SIZE = IMG_SIZE

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # label = self.df.label.values[idx]
        label = self.labels.values[idx]
        image = cv2.imread(self.images.values[idx])
        image = cv2.resize(image, dsize=self.IMG_SIZE)

        return image, label



args = Arguments()
df = read_csv(args.labels_csv, args.images_dir)
dataset = MyDataset(df)
dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

for images, labels in dataloader:
    print(images.shape, labels.shape)
