import os
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import scipy.misc as ssc

from psmnet.dataloader import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


calib = [[725.0087, 0, 620.5], [0, 725.0087, 187], [0, 0, 1]]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(root_dir, split):
    """
    Function to load data for Apollo
    :param root_dir: dataset directory
    :param split_file: file names
    :return: left, right and disparity file lists
    """
    if split == "train":
        scenes = ["Scene01", "Scene02", "Scene06", "Scene18"]
    else:
        scenes = ["Scene20"]

    sub_scenes = ["15-deg-left", "30-deg-left", "15-deg-right", "30-deg-right",
                  "clone", "morning", "rain", "fog", "overcast", "sunset"]

    left  = []
    right = []
    disp  = []

    for scene in scenes:
        dir = os.path.join(root_dir, scene)
        for sub in sub_scenes:
            sub_dir = os.path.join(dir, sub, "frames")
            path, dirs, files = os.walk(os.path.join(sub_dir, "rgb", "Camera_0")).__next__()
            num_files = len(files)

            for i in range(num_files):
                file = "{:05d}".format(i)
                left.append(os.path.join(sub_dir, "rgb", "Camera_0",
                                         "rgb_{}.jpg".format(file)))
                right.append(os.path.join(sub_dir, "rgb", "Camera_1",
                                         "rgb_{}.jpg".format(file)))
                disp.append(os.path.join(sub_dir, "depth", "Camera_0",
                                         "depth_{}.png".format(file)))

    return left, right, disp


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    depth = np.array(Image.open(path)).astype(np.float64) / 100.0 # convert to meters
    baseline = 0.54

    disparity = (1.5 * baseline * calib[0][0]) / (depth + 1e-6) # enhance disparity for better training
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
