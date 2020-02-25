#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
import os
import numpy as np
import scipy.misc as ssc
from kitti_util import *
import matplotlib.pyplot as plt


# In[122]:


disp_map = ssc.imread("../../vkitti_1.3.1_depthgt/0001/15-deg-left/00000.png")
# print(disp_map)
disp_map = (disp_map).astype(np.float32)/100
print(disp_map.max())

# disp_map = (disp_map*256).astype(np.uint16)/256
plt.figure(figsize=(10, 10))
plt.imshow(disp_map, cmap = 'gray')

f = open("/datasets/home/73/673/h6gupta/CSE291A/dataset/vkitti/vkitti_1.3.1_extrinsicsgt/0001_15-deg-left.txt", 'r')
x = f.readlines()[1]
x = x.split(" ")
R0 = np.array(x[1:4] + x[5:8] + x[9:12]).reshape([3,3]).astype('float32')

print(R0)


# In[123]:


# intrinsic = [[725.0087, 0, 620.5], [0, 725.0087, 187], [0, 0, 1]]
# a = [7.215377000000e+02, 0.000000000000e+00,6.095593000000e+02, 4.485728000000e+01,0.000000000000e+00,7.215377000000e+02,1.728540000000e+02,2.163791000000e-01,0.000000000000e+00,0.000000000000e+00,1.000000000000e+00,2.745884000000e-03]
# a = np.array(a).reshape([3,4])
# print(a[0, 3] / a[0, 0], a[1, 3] / a[1, 1])

a = [7.533745000000e-03,-9.999714000000e-01,-6.166020000000e-04,-4.069766000000e-03,1.480249000000e-02,7.280733000000e-04,-9.998902000000e-01,-7.631618000000e-02,9.998621000000e-01,7.523790000000e-03,1.480755000000e-02,-2.717806000000e-01]
a = np.array(a).reshape([3,4])
print(a)

def inverse_rigid_trans(Tr):
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

C2V = inverse_rigid_trans(a)
print(C2V)


# In[124]:



class Calibration(object):

    def __init__(self, R0, C2V):

        self.R0 = R0
        self.C2V = C2V
        
        self.P = np.array([725.0087, 0, 620.5, 0, 725.0087, 187, 0, 0, 1]).reshape([3,3])
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = -0.06
        self.b_y = -0.0029

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom
    

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))


    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # ===========================
    
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return pts_3d_rect
#         return self.project_rect_to_velo(pts_3d_rect)
  

calib = Calibration(R0, C2V)


# In[116]:


def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
#     print(cloud)
#     valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud

def project_disp_to_points(calib, disp, max_high):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]



lidar = project_depth_to_points(calib, disp_map, 1)

lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
lidar = lidar.astype(np.float32)

print(lidar)


# In[125]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

skip = 50 # plot one in every `skip` points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
velo_range = range(0, lidar.shape[0], skip) # skip points to prevent crash
ax.scatter(lidar[velo_range, 0],   
           lidar[velo_range, 1],
           lidar[velo_range, 2],   
           c=lidar[velo_range, 3],
           cmap='gray')
ax.set_title('Lidar scan (subsampled)')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[ ]:




