import numpy as np

from utils.container import G

__all__ = ['vkitti_attributes']


vkitti_attributes = G()
#vkitti_attributes.class_names = ('Car', 'Van', 'Truck')
# ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
#vkitti_attributes.class_name_to_size_template = {
#    'Car': np.array([3.85150801, 1.59570698, 1.50239394]),
#    'Van': np.array([4.66256793, 1.89853565, 2.02366792]),
#    'Truck': np.array([9.72038953, 2.67032881, 3.54183852])
#}

vkitti_attributes.class_names = ('Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc')
# ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
vkitti_attributes.class_name_to_size_template = {
    'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
    'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])
    }
