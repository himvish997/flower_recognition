import torch
import torchvision
import numpy
import pandas
import cv2
import efficientnet_pytorch
from torch.utils.data import dataloader, dataset



class Arguments():
    def __init__(self):
        self.labels_csv = 'path/to/labels.csv'
        self.images_dir = 'path/to/images_dir'

        self.batch_size = 16
        self.IMG_SIZE = (300, 300)

        self.cuda = False
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")
