import os
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import scipy.misc as ssc
import cv2

from psmnet.dataloader import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


calib = [[725.0087, 0, 620.5], [0, 725.0087, 187], [0, 0, 1]]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(root_dir, split_file):
    """
    Function to load data for Apollo
    :param root_dir: dataset directory
    :param split_file: file names
    :return: left, right and disparity file lists
    """
    files = pd.read_csv(split_file, header=0)

    left_train  = list(files["left"].apply(lambda x: root_dir + x))
    right_train = list(files["right"].apply(lambda x: root_dir + x))
    disp_train  = list(files["depth"].apply(lambda x: root_dir + x))

    return left_train, right_train, disp_train


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    depth = Image.open(path)
    baseline = 0.54

    disparity = (baseline * calib[0][0]) / (depth + 1e-6)
    return disparity


class ImageLoader(data.Dataset):
    def __init__(self, left, right, left_disparity, training,
                 loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

        else:
            w, h = left_img.size

            left_img = left_img.crop((w - 1200, h - 352, w, h))
            right_img = right_img.crop((w - 1200, h - 352, w, h))
            w1, h1 = left_img.size

            dataL = dataL[h - 352:h, w - 1200:w]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

        dataL = torch.from_numpy(dataL).float()
        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
