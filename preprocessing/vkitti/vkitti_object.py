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

scenes_dict = {"train": ["Scene01", "Scene02", "Scene06", "Scene18"],
               "val": ["Scene20"]}
sub_scenes = ["15-deg-left", "30-deg-left", "15-deg-right", "30-deg-right",
              "clone", "morning", "rain", "fog", "overcast", "sunset"]
MAX_DEPTH = 80


class vkitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split, scene, sub_scene):
        """root_dir contains scene folders"""
        self.root_dir = root_dir
        self.split = split
        # print(scene, scenes_dict[split])
        assert scene in scenes_dict[split]
        assert sub_scene in sub_scenes

        self.scene = scene
        self.sub_scene = sub_scene
        self.sub_scene_dir = os.path.join(self.root_dir, scene, sub_scene)
        self.image_dir = os.path.join(self.sub_scene_dir, "frames", "rgb")
        self.depth_dir = os.path.join(self.sub_scene_dir, "frames", "depth")
        self.cameras = ["Camera_0", "Camera_1"]
        self.intrinsic_file = os.path.join(self.sub_scene_dir, 'intrinsic.txt')
        self.extrinsic_file = os.path.join(self.sub_scene_dir, "extrinsic.txt")

        self.intrinsics = [self._process_file(self.intrinsic_file, 0),
                           self._process_file(self.intrinsic_file, 1)]

        self.extrinsics = [self._process_file(self.extrinsic_file, 0),
                           self._process_file(self.extrinsic_file, 1)]

        self.label_file_2d = os.path.join(self.sub_scene_dir, 'bbox.txt')
        self.label_file_3d = os.path.join(self.sub_scene_dir, 'pose.txt')
        self.label_object_type = os.path.join(self.sub_scene_dir, 'info.txt')
        self.labels = [self._get_label_data(0), self._get_label_data(1)]

        path, dirs, files = next(
            os.walk(os.path.join(self.image_dir, self.cameras[0])))
        self.num_samples = len(files)

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _process_file(file, cam_idx):
        params = pd.read_csv(file, sep=" ", header=0)
        return params[params["cameraID"] == cam_idx]

    def get_image(self, idx, cam_idx):
        assert (idx < self.num_samples)
        image_dir = os.path.join(self.image_dir, self.cameras[cam_idx])
        img_filename = os.path.join(image_dir, "rgb_{:05d}.jpg".format(idx))
        return utils.load_image(img_filename)

    def get_calibration(self, idx, cam_idx):
        assert (idx < self.num_samples)
        return utils.Calibration(self.scene, self.sub_scene,
                                 self.intrinsics[cam_idx].iloc[idx],
                                 self.extrinsics[cam_idx].iloc[idx])

    def get_label_objects(self, idx, cam_idx):
        assert (idx < self.num_samples)
        labels = self.labels[cam_idx]
        label_data = labels[labels["frame"] == idx]
        objects = [utils.Object3d(row) for idx, row in label_data.iterrows()]
        return objects

    def get_depth_map(self, idx, cam_idx):
        assert (idx < self.num_samples)
        depth_dir = os.path.join(self.depth_dir, self.cameras[cam_idx])
        filename = os.path.join(depth_dir, "depth_{:05d}.png".format(idx))
        return utils.load_depth(filename)

    def _get_label_data(self, cam_idx):
        bbox = pd.read_csv(self.label_file_2d, sep=" ", header=0)
        obj = pd.read_csv(self.label_file_3d, sep=" ", header=0)
        data = pd.merge(bbox, obj, on=["frame", "cameraID", "trackID"],
                        how="inner")
        data = data[data["cameraID"] == cam_idx]

        info = pd.read_csv(self.label_object_type, sep=" ", header=0)
        data = pd.merge(data, info, on="trackID")
        return data

    def get_lidar(self, idx, cam_idx):
        calib = self.get_calibration(idx, cam_idx)
        depth = self.get_depth_map(idx, cam_idx)
        velo = project_depth_to_points(calib, depth)
        velo = np.concatenate([velo, np.ones((velo.shape[0], 1))], 1)
        return velo


def project_depth_to_points(calib, depth, max_high=1.0):
    depth[depth > MAX_DEPTH] = -1
    depth = np.round(depth, 2)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    mask = (depth > 0) & (c % 2 == 0) & (r % 2 == 0)
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    # print(np.max(depth), np.min(depth))
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


def show_image_with_boxes(img, objects, calib, show3d=True):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == 'DontCare': continue
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), (0, 255, 0), 2)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
    Image.fromarray(img1).show()
    if show3d:
        Image.fromarray(img2).show()


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def show_lidar_with_boxes(pc_velo, objects, calib,
                          img_fov=False, img_width=None, img_height=None):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
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
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
        mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5),
                    tube_radius=None, line_width=1, figure=fig)
    # mlab.show(1)
    return fig


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
                                                              calib, 0, 0,
                                                              img_width,
                                                              img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i, 0])),
                         int(np.round(imgfov_pts_2d[i, 1]))),
                   2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show()
    return img
