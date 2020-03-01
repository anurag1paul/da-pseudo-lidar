''' Helper class and functions for loading Virtual KITTI objects

Author: Anurag Paul
Date: February 2020
'''
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import vkitti.vkitti_util as utils

raw_input = input  # Python 3

sub_scenes = ["15-deg-left", "30-deg-left", "15-deg-right", "30-deg-right",
              "clone", "morning", "rain", "fog", "overcast", "sunset"]


class vkitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split, scene, sub_scene):
        """root_dir contains scene folders"""
        self.root_dir = root_dir
        self.split = split
        if split == "train":
            scenes = ["Scene01", "Scene02", "Scene06", "Scene18"]
        else:
            scenes = ["Scene20"]

        assert scene in scenes
        assert sub_scene in sub_scenes

        self.sub_scene_dir = os.path.join(self.root_dir, scene, sub_scene)
        self.image_dir = os.path.join(self.sub_scene_dir, "frames", "rgb", "Camera_0")
        self.depth_dir = os.path.join(self.sub_scene_dir, "frames", "depth", "Camera_0")

        self.intrinsic_file = os.path.join(self.sub_scene_dir, 'intrinsic.txt')
        self.extrinsic_file = os.path.join(self.sub_scene_dir, "extrinsic.txt")

        self.intrinsics = self._process_file(self.intrinsic_file)
        self.extrinsics = self._process_file(self.extrinsic_file)

        self.label_file_2d = os.path.join(self.sub_scene_dir, 'bbox.txt')
        self.label_file_3d = os.path.join(self.sub_scene_dir, 'pose.txt')
        self.label_object_type = os.path.join(self.sub_scene_dir, 'info.txt')
        self.labels = self._get_label_data()

        path, dirs, files = next(os.walk(self.image_dir))
        self.num_samples = len(files)

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _process_file(file):
        params = pd.read_csv(file, sep=" ", header=0)
        return params[params["cameraID"] == 0]

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, "rgb_{:05d}.jpg".format(idx))
        return utils.load_image(img_filename)

    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        return utils.Calibration(self.intrinsics.iloc[idx],
                                 self.extrinsics.iloc[idx])

    def get_label_objects(self, idx):
        assert (idx < self.num_samples)
        label_data = self.labels[self.labels["frame"] == idx]
        objects = [utils.Object3d(row) for idx, row in label_data.iterrows()]
        return objects

    def get_depth_map(self, idx):
        assert (idx < self.num_samples)
        filename = os.path.join(self.depth_dir, "depth_{:05d}.png".format(idx))
        return utils.load_depth(filename)

    def _get_label_data(self):
        bbox = pd.read_csv(self.label_file_2d, sep=" ", header=0)
        obj = pd.read_csv(self.label_file_3d, sep=" ", header=0)
        data = pd.merge(bbox, obj, on=["frame", "cameraID", "trackID"], how="inner")
        data = data[data["cameraID"] == 0]

        info = pd.read_csv(self.label_object_type, sep=" ", header=0)
        data = pd.merge(data, info, on="trackID")
        return data

    def get_cloud(self, idx):
        calib = self.get_calibration(idx)
        depth = self.get_depth_map(idx)
        velo = project_depth_to_points(calib, depth)
        velo = np.concatenate([velo, np.ones((velo.shape[0], 1))], 1)
        return velo


def project_depth_to_points(calib, depth, max_high=3.0):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_ref(points)
    # valid = (cloud[:, 2] < max_high)
    return cloud


def show_image_with_boxes(img, objects, calib, show3d=True):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == 'DontCare': continue
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), (0, 255, 0), 2)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        print(box3d_pts_2d)
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
    Image.fromarray(img1).show()
    if show3d:
        Image.fromarray(img2).show()


def get_lidar_in_image_fov(pc_rect, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_ref_to_image(pc_rect)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    # fov_inds = fov_inds & (pc_rect[:, 0] > clip_distance)
    imgfov_pc_velo = pc_rect[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def show_lidar_with_boxes(pc_velo, objects, calib,
                          img_fov=False, img_width=None, img_height=None):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from vkitti.viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
                                         img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))

    draw_lidar(pc_velo, fig=fig)

    for obj in objects:
        if obj.type == 'DontCare': continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_ref(box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_ref(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
        mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5),
                    tube_radius=None, line_width=1, figure=fig)
    mlab.show(1)

