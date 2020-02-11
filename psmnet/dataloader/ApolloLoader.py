# 3130 x 960
import os
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from psmnet.dataloader import preprocess


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(file_paths):
    """
    Function to load data for Apollo
    :param file_paths: list of training folders
    :return: left, right and disparity file lists
    """
    left = 'camera_5/'
    right = 'camera_6/'
    disp = 'disparity/'

    left_train = []
    right_train = []
    disp_train = []

    for filepath in file_paths:
        left_folder = os.path.join(filepath, left)
        right_folder = os.path.join(filepath, right)
        disp_folder = os.path.join(filepath, disp)

        left_files = next(os.walk(left_folder))[2]

        for left_img in left_files:
            left_train.append(os.path.join(left_folder, left_img))
            right_train.append(os.path.join(right_folder, left_img.replace("Camera_5", "Camera_6")))
            disp_train.append(os.path.join(disp_folder, left_img.replace("jpg", "png")))

    return left_train, right_train, disp_train


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    disp_img = Image.open(path)
    return np.array(disp_img).astype(np.float32)


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
            dataL = dataL[h - 352:h, w - 1200:w]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

        dataL = torch.from_numpy(dataL).float()
        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
