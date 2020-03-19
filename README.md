# Domain Adaptation for 3D Object Detection
This project was submitted as part of the requirements of the course CSE 291 A 
(Domain Adaptation) by Prof. Manmohan Chandraker at UC San Diego. 

## Contents

- [Introduction](#introduction)
- [Code Structure](#2.-code-structure)
- [Usage](#usage)
- [Contacts](#contacts)

## Introduction
3D object detection is an important component of autonomous driving. 
However, LiDAR sensors are expensive and so is labeling large amount of data. 
Thus, in this project, we try to use domain adaptation to solve 3D object 
detection problem using only images at the time of inference.

## Usage

### 1. Overview

The project basically has three components:
- Data Processing: mainly contained in preprocessing folder and uses Frustum-Pointnets repo
- Depth Adaptation: This is done using T2Net and code is contained in Synthetic2Realistic submodule
- 3D Object Detection DA: frustum_pointnet folder contains all the relevant code  

### 2. Code Structure

#### 2.1 Preprocessing
- generate_vkitti_frustum.py: generates frustums for Virtual KITTI data
- generate_lidar.py: generates pseudo-LiDAR based on KITTI depth predictions 

#### 2.2 Synthetic2Realistic
- train.py: for training the model
- test.py: for generating the depth predictions using the model
- model: contains all the relevant model details
- dataloader: creates file or directory based dataloaders
- options: specifies the list of all available training and testing options

#### 2.3 frustum_pointnet
- configs: contains options for each experiment
- data: stores frustum data for Virtual KITTI and KITTI
- datasets: dataloaders
- evaluate: code for evaluating performance
- meters: code for evaluation metrics
- models: models details
- modules: losses and C++ pytorch modules

Training files:
- train.py: trains target only model using Frustum-PointNet
- train_vkitti.py: trains source only model using Frustum-PointNet
- train_dan.py: trains Frustum-PointDAN Basic
- train_dan_full.py: trains PointDAN for all the sub-models of Frustum-PointNet
- train_dan_parallel.py: trains Frustum-PointDAN Parallel
- train_dan_simple.py: trains DA Frustum-PointNet Basic
- train_dan_simple_parallel.py: trains DA Frustum-PointNet Parallel
- train_dan_simple_full.py: trains DA Frustum-PointNet for all three sub-models

### 3. Usage

#### 3.1 Preprocessing
- Generate Vkitti Frustums
```buildoutcfg
cd preprocessing
python3 generate_vkitti_frustum.py --gen_train --gen_val --car_only 
--path <path to vkitti folder>
```

#### 3.2 Depth Estimation
```buildoutcfg
git submodule init
git submodule update
cd Synthetic2Realistic
python3 train.py <specify options>
```
Please look at mytrain.sh and mytest.sh for suggested use.

#### 3.3 3D Object Detection
```buildoutcfg
cd frustum_pointnet
sh setup.sh
python3 <train_file> --configs <specify config file> --devices <GPU IDs>
``` 
- Training Example
```buildoutcfg
python3 train_dan_simple.py --configs configs/dan/simple/simpledan.py --devices 0
```

- Testing Example
```buildoutcfg
python3 train_dan_simple.py --configs configs/dan/simple/simpledan.py --devices 0 --evaluate
```

## Contact
If you have any question, please feel free to email us.

Anurag Paul (anurag1paul@gmail.com), Manjot Singh Bilkhu, Devendra Partap Yadav, Harshul Gupta 
