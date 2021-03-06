''' Prepare Virtual KITTI data for 3D object detection.

Date: February 2020
'''

from __future__ import print_function

import pickle
import argparse

from collections import defaultdict

from vkitti.vkitti_object import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds


def demo(path):
    import mayavi.mlab as mlab
    from vkitti.viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = vkitti_object(path, "train", "Scene06", "clone")
    data_idx = 50
    cam_idx = 0

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx, cam_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx, cam_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    pc_velo = dataset.get_lidar(data_idx, cam_idx)[:,0:3]
    print("Velo: ", pc_velo.shape)
    calib = dataset.get_calibration(data_idx, cam_idx)

    ## Draw lidar in rect camera coord
    #print(' -------- LiDAR points in rect camera coordination --------')
    #pc_rect = calib.project_velo_to_rect(pc_velo)
    #fig = draw_lidar_simple(pc_rect)
    #raw_input()

    # Draw 2d and 3d boxes on image
    print(' -------- 2D/3D bounding boxes in images --------')
    show_image_with_boxes(img, objects, calib)
    raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    #show_lidar_with_boxes(pc_velo, objects, calib)
    #raw_input()
    fig = show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    mlab.savefig("vkitti_velo_rot.png", figure=fig, magnification=6.0)
    raw_input()

    # Visualize LiDAR points on images
    # print(' -------- LiDAR points projected to image plane --------')
    # show_lidar_on_image(pc_velo, img, calib, img_width, img_height)
    # raw_input()

    # Show LiDAR points that are in the 3d box
    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    print(box3d_pts_3d_velo)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()

    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:,0:2] = imgfov_pts_2d
    cameraUVDepth[:,2] = imgfov_pc_rect[:,2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    xmin,ymin,xmax,ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()


def test(path, scene):
    import mayavi.mlab as mlab
    from vkitti.viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    for sub_scene in sub_scenes[:5]:
        print(sub_scene)
        dataset = vkitti_object(path, "train", scene, sub_scene)
        data_idx = 25
        cam_idx = 0

        # Load data from dataset
        objects = dataset.get_label_objects(data_idx, cam_idx)
        objects[0].print_object()
        img = dataset.get_image(data_idx, cam_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        print(('Image shape: ', img.shape))
        pc_velo = dataset.get_lidar(data_idx, cam_idx)[:, 0:3]
        calib = dataset.get_calibration(data_idx, cam_idx)

        fig = show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
        mlab.savefig('{}_{}.jpg'.format(scene, sub_scene), figure=fig)


def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height '''
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx2 = cx + w * r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)
    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    return np.array(
        [cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])


def extract_frustum_data(path, split, output_filename, viz=False,
                         perturb_box2d=False, augmentX=1,
                         type_whitelist=['Car']):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either training or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    cam_idx = 0

    for scene in scenes_dict[split]:
        id_list = []  # identifier
        box2d_list = []  # [xmin, ymin, xmax, ymax]
        box3d_list = []  # (8,3) array in rect camera coord
        input_list = []  # channel number = 4, xyz,intensity in rect camera coord
        label_list = []  # 1 for roi object, 0 for clutter
        type_list = []  # string e.g. Car
        heading_list = []  # ry (along y-axis in rect camera coord) radius of
        # (cont.) clockwise angle from positive x axis in velo coord.
        box3d_size_list = []  # array of l,w,h
        frustum_angle_list = []  # angle of 2d box center from pos x-axis
        rejected = 0

        for sub_scene in sub_scenes:
            dataset = vkitti_object(path, split, scene, sub_scene)
            pos_cnt = 0
            all_cnt = 0
            for data_idx in range(len(dataset)):
                idx = "{}/{}/{}".format(scene, sub_scene, data_idx)
                print('------------- ', idx)

                calib = dataset.get_calibration(data_idx, cam_idx) # 3 by 4 matrix
                objects = dataset.get_label_objects(data_idx, cam_idx)
                pc_velo = dataset.get_lidar(data_idx, cam_idx)
                pc_rect = np.zeros_like(pc_velo)
                pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
                pc_rect[:,3] = pc_velo[:,3]
                img = dataset.get_image(data_idx, cam_idx)
                img_height, img_width, img_channel = img.shape
                _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
                calib, 0, 0, img_width, img_height, True)

                for obj_idx in range(len(objects)):
                    if objects[obj_idx].type not in type_whitelist :continue

                    # 2D BOX: Get pts rect backprojected
                    box2d = objects[obj_idx].box2d
                    for _ in range(augmentX):
                        # Augment data by box2d perturbation
                        if perturb_box2d:
                            xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                        else:
                            xmin, ymin, xmax, ymax = box2d
                        box_fov_inds = (pc_image_coord[:,0]<xmax) & \
                            (pc_image_coord[:,0]>=xmin) & \
                            (pc_image_coord[:,1]<ymax) & \
                            (pc_image_coord[:,1]>=ymin)
                        box_fov_inds = box_fov_inds & img_fov_inds
                        pc_in_box_fov = pc_rect[box_fov_inds,:]
                        # Get frustum angle (according to center pixel in 2D BOX)
                        box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                        uvdepth = np.zeros((1,3))
                        uvdepth[0,0:2] = box2d_center
                        uvdepth[0,2] = 20 # some random depth
                        box2d_center_rect = calib.project_image_to_rect(uvdepth)
                        frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
                            box2d_center_rect[0,0])
                        # 3D BOX: Get pts velo in 3d box
                        obj = objects[obj_idx]
                        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                        _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                        label = np.zeros((pc_in_box_fov.shape[0]))
                        label[inds] = 1
                        # Get 3D BOX heading
                        heading_angle = obj.ry
                        # Get 3D BOX size
                        box3d_size = np.array([obj.l, obj.w, obj.h])

                        # Reject too far away object or object without points
                        if ymax-ymin<25 or np.sum(label)==0:
                            rejected += 1
                            continue

                        id_list.append(data_idx)
                        box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                        box3d_list.append(box3d_pts_3d)
                        input_list.append(pc_in_box_fov)
                        label_list.append(label)
                        type_list.append(objects[obj_idx].type)
                        heading_list.append(heading_angle)
                        box3d_size_list.append(box3d_size)
                        frustum_angle_list.append(frustum_angle)

                        # collect statistics
                        pos_cnt += np.sum(label)
                        all_cnt += pc_in_box_fov.shape[0]

        print("Number of boxes:{}, rejected:{}".format(len(input_list), rejected))
        print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
        print('Average npoints: %f' % (float(all_cnt) / len(id_list)))

        with open("{}_{}.pickle".format(output_filename, scene), 'wb') as fp:
            pickle.dump(id_list, fp)
            pickle.dump(box2d_list, fp)
            pickle.dump(box3d_list, fp)
            pickle.dump(input_list, fp)
            pickle.dump(label_list, fp)
            pickle.dump(type_list, fp)
            pickle.dump(heading_list, fp)
            pickle.dump(box3d_size_list, fp)
            pickle.dump(frustum_angle_list, fp)

    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            seg = label_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
                              fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:, 0], p1[:, 1], p1[:, 2], seg, mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
                              fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:, 2], -p1[:, 0], -p1[:, 1], seg, mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()


def get_box3d_dim_statistics(path):
    ''' Collect and dump 3D bounding box statistics '''
    split = "train"
    dimension_list = defaultdict(list)
    
    for scene in scenes_dict[split]:
        for sub_scene in sub_scenes:
            dataset = vkitti_object(path, split, scene, sub_scene)
            for data_idx in range(len(dataset)):
                idx = "{}/{}/{}".format(scene, sub_scene, data_idx)
                print('------------- ', idx)
                objects = dataset.get_label_objects(data_idx, cam_idx=0)
                for obj_idx in range(len(objects)):
                    obj = objects[obj_idx]
                    if obj.type=='DontCare':continue
                    dimension_list[obj.type].append(np.array([obj.l,obj.w,obj.h]))
    
    for type, dims in dimension_list.items():
        dims_data = np.array(dims)
        print(type, dims_data.shape)
        print(type, dims_data.mean(axis=0))


def write_gt_file(val_folder, tot_idx, objects):
    path = os.path.join(val_folder, "{:06d}.txt".format(tot_idx))
    with open(path, "w") as f:
        for i in range(len(objects)):
            obj = objects[i]
            row = "{} {} {} {} ".format(obj.type, obj.truncation, obj.occlusion, obj.alpha)
            row += "{} {} {} {} ".format(*obj.box2d)
            row += "{} {} {} ".format(obj.h, obj.w, obj.l)
            row += "{} {} {} ".format(*obj.t)
            row += "{}\n".format(obj.ry)
            f.write(row)


def extract_frustum_data_rgb_detection(path, split, output_filename,
                                       viz=False,
                                       type_whitelist=['Car'],
                                       img_height_threshold=25,
                                       lidar_point_threshold=5):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''
    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = [] # angle of 2d box center from pos x-axis
    r = 0
    cam_idx = 0
    tot_idx = 0

    val_folder = "val"
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    for scene in scenes_dict[split]:
        for sub_scene in sub_scenes:
            dataset = vkitti_object(path, split, scene, sub_scene)
            for data_idx in range(len(dataset)):
                idx = "{}/{}/{}".format(scene, sub_scene, data_idx)
                print('------------- ', idx)

                calib = dataset.get_calibration(data_idx, cam_idx) # 3 by 4 matrix
                pc_velo = dataset.get_lidar(data_idx, cam_idx)
                pc_rect = np.zeros_like(pc_velo)
                pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
                pc_rect[:,3] = pc_velo[:,3]
                img = dataset.get_image(data_idx, cam_idx)
                img_height, img_width, img_channel = img.shape
                _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(
                    pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)

                objects = dataset.get_label_objects(data_idx, cam_idx)
                gt_objects = []

                for det_idx in range(len(objects)):
                    if objects[det_idx].type not in type_whitelist :continue

                    # 2D BOX: Get pts rect backprojected
                    xmin,ymin,xmax,ymax = objects[det_idx].box2d
                    box_fov_inds = (pc_image_coord[:,0] < xmax) & \
                        (pc_image_coord[:,0] >= xmin) & \
                        (pc_image_coord[:,1] < ymax) & \
                        (pc_image_coord[:,1] >= ymin)
                    box_fov_inds = box_fov_inds & img_fov_inds
                    pc_in_box_fov = pc_rect[box_fov_inds,:]
                    # Get frustum angle (according to center pixel in 2D BOX)
                    box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                    uvdepth = np.zeros((1,3))
                    uvdepth[0, 0:2] = box2d_center
                    uvdepth[0, 2] = 20 # some random depth
                    box2d_center_rect = calib.project_image_to_rect(uvdepth)
                    frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
                        box2d_center_rect[0,0])

                    # Pass objects that are too small
                    if ymax-ymin<img_height_threshold or \
                        len(pc_in_box_fov)<lidar_point_threshold:
                        continue

                    id_list.append(tot_idx)
                    type_list.append(objects[det_idx].type)
                    box2d_list.append(objects[det_idx].box2d)
                    prob_list.append(1.0)
                    input_list.append(pc_in_box_fov)
                    frustum_angle_list.append(frustum_angle)
                    gt_objects.append(objects[det_idx])
                write_gt_file(val_folder, tot_idx, gt_objects)
                tot_idx += 1

    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)

    with open("val.txt", "w") as f:
        for i in range(tot_idx):
            f.write("{:06d}\n".format(i))

    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], p1[:,1], mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--test', action='store_true', help='Run test.')
    parser.add_argument('--path', help='Vkitti data path')
    parser.add_argument('--gen_train', action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true',
                        help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true',
                        help='Only generate cars; otherwise cars, peds and cycs')
    parser.add_argument('--stats', action='store_true',
                        help='generate 3d stats')
    args = parser.parse_args()

    if args.test:
        test(args.path, "Scene18")

    if args.demo:
        demo(args.path)
        exit()

    if args.stats:
        get_box3d_dim_statistics(args.path)

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Van', 'CTruck']
        output_prefix = 'frustum_carvantruck_'

    if args.gen_train:
        extract_frustum_data(
            args.path,
            'train',
            os.path.join(BASE_DIR, output_prefix + 'train'),
            viz=False, perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist)

    if args.gen_val:
        extract_frustum_data(
            args.path,
            'val',
            os.path.join(BASE_DIR, output_prefix + 'val'),
            viz=False, perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist)

    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection(
            args.path,
            'val',
            os.path.join(BASE_DIR, output_prefix+'val_rgb_detection.pickle'),
            viz=False,
            type_whitelist=type_whitelist)
