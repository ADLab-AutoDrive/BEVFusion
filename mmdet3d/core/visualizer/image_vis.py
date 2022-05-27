# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from matplotlib import pyplot as plt


def translate(points: np.ndarray, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            points[i, :] = points[i, :] + x[i]

def rotate(points: np.ndarray, rot_matrix: np.ndarray) -> None:
    """
    Applies a rotation.
    :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
    """
    points[:3, :] = np.dot(rot_matrix, points[:3, :])

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        #TODO: check why have zeros value here
        z = points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
        points = points / (z + 1e-10)

    return points

def vis_depth(img, depth, name):
    from matplotlib import pyplot as plt
    import cv2
    h, w = depth.shape
    x = []
    y = []
    for i in range(h):
        x.append(np.ones(w) * i)
        y.append(np.arange(w))
    x = np.concatenate(x)
    y = np.concatenate(y)
    img= ((img.permute(1,2,0)+3)/6).cpu().numpy().astype(np.float32)
    if img.shape[0]!=h:
        img = cv2.resize(img, (w, h))
    depth = depth.view(-1).cpu().numpy()
    mask = depth>0
    x = x[mask].astype(np.int32)
    y = y[mask].astype(np.int32)
    depth = depth[mask]
    fig, ax = plt.subplots()
    ax.imshow(img.astype(np.float32))
    ax.scatter(y, x, c=depth, s=1)
    ax.axis('off')
    fig.savefig('work_dirs/img_{}.png'.format(name),bbox_inches='tight',pad_inches = 0)

def map_pointcloud_to_image(points, 
                                img,
                                sensor2lidar_r,
                                sensor2lidar_t,
                                camera_intrinsic,
                                show=False):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    points = copy.deepcopy(points.T)
    translate(points, -sensor2lidar_t)
    rotate(points, sensor2lidar_r.T)
    
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = points[2, :]
    coloring = depths
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(points[:3, :], camera_intrinsic, normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    depth_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    xs = np.minimum((points[0,:] + 0.5).astype(np.int32), depth_map.shape[1])
    ys = np.minimum((points[1,:] + 0.5).astype(np.int32), depth_map.shape[0])
    for x, y, c in zip(xs, ys, coloring):
        depth_map[y, x] = c
    if show:
        if not os.path.exists('work_dirs'):
            os.mkdir('work_dirs')
        plt.imsave('work_dirs/depth_map.png', depth_map, dpi=1)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.scatter(points[0, :], points[1, :], c=coloring, s=1)
        ax.axis('off')
        fig.savefig('work_dirs/img_depth.png',bbox_inches='tight',pad_inches = 0)
    return depth_map


def project_pts_on_img(points,
                       raw_img,
                       lidar2img_rt,
                       max_distance=70,
                       thickness=-1):
    """Project the 3D points cloud on 2D image.

    Args:
        points (numpy.array): 3D points cloud (x, y, z) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        max_distance (float, optional): the max distance of the points cloud.
            Default: 70.
        thickness (int, optional): The thickness of 2D points. Default: -1.
    """
    img = raw_img.copy()
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    fov_inds = ((pts_2d[:, 0] < img.shape[1])
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img.shape[0])
                & (pts_2d[:, 1] >= 0))

    imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pts_2d[i, 2]
        color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=1,
            color=tuple(color),
            thickness=thickness,
        )
    cv2.imwrite('work_dirs/project_pts_img.png', img.astype(np.uint8))
    # cv2.imshow('project_pts_img', img.astype(np.uint8))
    # cv2.waitKey(100)


def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


# TODO: remove third parameter in all functions here in favour of img_metas
def draw_depth_bbox3d_on_img(bboxes3d,
                             raw_img,
                             calibs,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`DepthInstance3DBoxes`, shape=[M, 7]):
            3d bbox in depth coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        calibs (dict): Camera calibration information, Rt and K.
        img_metas (dict): Used in coordinates transformation.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.bbox import points_cam2img
    from mmdet3d.models import apply_3d_transformation

    img = raw_img.copy()
    img_metas = copy.deepcopy(img_metas)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)

    # first reverse the data transformations
    xyz_depth = apply_3d_transformation(
        points_3d, 'DEPTH', img_metas, reverse=True)

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(xyz_depth,
                               xyz_depth.new_tensor(img_metas['depth2img']))
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_camera_bbox3d_on_img(bboxes3d,
                              raw_img,
                              cam2img,
                              img_metas,
                              color=(0, 255, 0),
                              thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]):
            3d bbox in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.bbox import points_cam2img

    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)