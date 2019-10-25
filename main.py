import torch
import torchvision
import numpy as np
import os
import pandas as pd
import cv2
import efficientnet_pytorch
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split

from utils.image import display
from utils.print_function import print_function


class Arguments():
    def __init__(self):
        self.labels_csv = '/mnt/NewVolume/projects/flower_recognition/data/train.csv'
        self.images_dir = '/mnt/NewVolume/projects/flower_recognition/data/train'
        self.train_val_split = 0.05
        # Learning Rate
        self.lr = 0.001
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
    def __init__(self, dataframe, IMG_SIZE, transform=None):
        self.labels = dataframe['labels']
        self.images = dataframe['images']

        self.IMG_SIZE = IMG_SIZE
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # label = self.df.label.values[idx]
        label = self.labels.values[idx]
        image = cv2.imread(self.images.values[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.IMG_SIZE)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)
        image = torchvision.transforms.ToPILImage()(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label



args = Arguments()
df = read_csv(args.labels_csv, args.images_dir)
train_df, test_df = train_test_split(df, test_size=args.train_val_split, shuffle=True, stratify=df['labels'])
train_df, test_df = train_test_split(test_df, test_size=0.2, shuffle=True, stratify=test_df['labels'])


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation((-120, 120)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = MyDataset(train_df, args.IMG_SIZE, train_transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

test_transform = torchvision.transforms.Compose([
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomRotation((-120, 120)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_dataset = MyDataset(test_df, args.IMG_SIZE, test_transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

model = EfficientNet.from_pretrained(model_name='efficientnet-b0', num_classes=102)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
loss_function = torch.nn.CrossEntropyLoss()

model.to(args.device)
optimizer.zero_grad()


import random
import time




def get_accuracy(output, labels):
    labels = labels.view((len(labels), 1))
    prob, cls = output.topk(1, dim=-1)
    equals = cls == labels
    return torch.mean(equals.type(torch.FloatTensor)).numpy()


def get_metrics():
    return {'acc': 0,
            'loss': 0,
            'val_loss': 0,
            'val_acc': 0}


def learning(train_data, val_data):
    metrics = get_metrics()
    running_metrics = get_metrics()

    model.train()
    total_step = len(train_data)
    for step, (image, label) in enumerate(train_data, 1):
        image = image.to(args.device)
        label = label.to(args.device)
        output =  model(image)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Metrics
        metrics['acc'] += get_accuracy(output=output, labels=label)
        running_metrics['acc'] = metrics['acc'] / step
        metrics['loss'] += loss.item()
        running_metrics['loss'] = metrics['loss'] / step
        print_function(metrics=running_metrics, is_progress=True, step=step, total_steps=total_step)
    model.eval()
    total_step = len(val_data)
    for step, (image, label) in enumerate(val_data, 1):
        image = image.to(args.device)
        label = label.to(args.device)
        output = model(image)
        metrics['val_loss'] += loss_function(output, label)
        running_metrics['val_loss'] = metrics['val_loss'] / step
        metrics['val_acc'] += get_accuracy(output=output, labels=label)
        running_metrics['val_acc'] = metrics['val_acc'] / step
        print_function(metrics=running_metrics, is_progress=True, step=step, total_steps=total_step)

    return metrics

learning(train_data=train_dataloader, val_data=test_dataloader)


