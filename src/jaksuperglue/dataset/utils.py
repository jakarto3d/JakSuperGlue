from typing import Tuple
from pyocamcalib.modelling.camera import Camera, cartesian2geographic
import numpy as np


def points_fisheye2equirectangular(uv_points: np.array,
                                   downsampling: int,
                                   camera: Camera,
                                   rotation_matrix: np.array):
    # Default equirectangular specifications
    he, we = (4096, 8192)
    he_new, we_new = he // downsampling, we // downsampling

    xyz = camera.cam2world(uv_points.astype(float))
    xyz_rotated = xyz @ rotation_matrix.T
    longitude, latitude = cartesian2geographic(xyz_rotated)
    x = longitude / np.pi
    y = latitude / (np.pi / 2)
    ve = we_new * (x + 1) / 2
    ue = he_new * (1 - y) / 2
    uv_points_e = np.vstack((ue, ve)).T
    uv_points_e = np.round(uv_points_e).astype(int)
    cond_1 = (uv_points_e[:, 0] < he_new) & (uv_points_e[:, 0] >= 0)
    cond_2 = (uv_points_e[:, 1] < we_new) & (uv_points_e[:, 1] >= 0)
    mask = cond_1 & cond_2

    return uv_points[mask, :], uv_points_e[mask, :]

def fisheye_1_to_fb_d_1(uv_points: np.array,
                        downsampling: int,
                        patch_size: Tuple[int, int],
                        camera: Camera,
                        rotation_matrix: np.array):


    # Default equirectangular specifications
    he, we = (4096, 8192)
    he_new, we_new = he // downsampling, we // downsampling

    # Replace points in original fisheye camera frame 1
    hf, wf = (2160, 3840)
    sub_fb_shape = (5 * patch_size[0], 10 * patch_size[1])
    f_ratio_h = sub_fb_shape[0] / hf
    f_ratio_w = sub_fb_shape[1] / wf
    vu_points_fisheye = np.vstack((np.round(uv_points[:, 1] / f_ratio_w),
                                   np.round(uv_points[:, 0] / f_ratio_h))).T

    vu_points, uv_points_equi = points_fisheye2equirectangular(vu_points_fisheye, downsampling, camera, rotation_matrix)

    # Replace fisheye in down sampled frame
    uv_points = vu_points[:, ::-1]
    uv_points[:, 0] *= f_ratio_h
    uv_points[:, 1] *= f_ratio_w

    # Replace equirectangular points in sub frame fb_d
    uv_sub_fb_d = np.copy(uv_points_equi)
    uv_sub_fb_d[:, 1] = (uv_sub_fb_d[:, 1] - patch_size[0] * 7) % we_new
    uv_sub_fb_d[:, 0] = (uv_sub_fb_d[:, 0] - patch_size[1]) % he_new

    # Replace equirectangular points in sub frame 1, i.e. fb_d_1
    mask_1 = (uv_sub_fb_d[:, 0] >= 0) & (uv_sub_fb_d[:, 0] < sub_fb_shape[0])
    mask_2 = (uv_sub_fb_d[:, 1] >= 0) & (uv_sub_fb_d[:, 1] < (sub_fb_shape[1] // 2))
    mask = mask_1 & mask_2


    return uv_points[mask, :], uv_sub_fb_d[mask, :]


def fisheye_2_to_fb_d_2(uv_points: np.array,
                        downsampling: int,
                        patch_size: Tuple[int, int],
                        camera: Camera,
                        rotation_matrix: np.array):

    # Default equirectangular specifications
    he, we = (4096, 8192)
    he_new, we_new = he // downsampling, we // downsampling

    # Replace points in original fisheye camera frame 2
    hf, wf = (2160, 3840)
    sub_fb_shape = (5 * patch_size[0], 10 * patch_size[1])
    f_ratio_h = sub_fb_shape[0] / hf
    f_ratio_w = sub_fb_shape[1] / wf
    vu_points_fisheye = np.vstack((np.round((uv_points[:, 1] + sub_fb_shape[1] / 2) / f_ratio_w),
                                   np.round(uv_points[:, 0] / f_ratio_h))).T

    vu_points, uv_points_equi = points_fisheye2equirectangular(vu_points_fisheye, downsampling, camera, rotation_matrix)

    # Replace fisheye in down sampled frame
    uv_points = vu_points[:, ::-1]
    uv_points[:, 0] *= f_ratio_h
    uv_points[:, 1] *= f_ratio_w
    uv_points[:, 1] -= sub_fb_shape[1] // 2

    # Replace equirectangular points in sub frame fb_d
    uv_sub_fb_d = np.copy(uv_points_equi)
    uv_sub_fb_d[:, 1] = (uv_sub_fb_d[:, 1] - patch_size[0] * 7) % we_new
    uv_sub_fb_d[:, 0] = (uv_sub_fb_d[:, 0] - patch_size[1]) % he_new

    # Replace equirectangular points in sub frame 2, i.e. fb_d_2
    mask_1 = (uv_sub_fb_d[:, 0] >= 0) & (uv_sub_fb_d[:, 0] < sub_fb_shape[0])
    mask_2 = (uv_sub_fb_d[:, 1] >= sub_fb_shape[1] // 2) & (uv_sub_fb_d[:, 1] < sub_fb_shape[1])
    mask = mask_1 & mask_2

    uv_sub_fb_d[:, 1] -= sub_fb_shape[1] // 2

    return uv_points[mask, :], uv_sub_fb_d[mask, :]


def fisheye_1_to_fb_g_1(uv_points: np.array,
                        downsampling: int,
                        patch_size: Tuple[int, int],
                        camera: Camera,
                        rotation_matrix: np.array):


    # Default equirectangular specifications
    he, we = (4096, 8192)
    he_new, we_new = he // downsampling, we // downsampling

    # Replace points in original fisheye camera frame 1
    hf, wf = (2160, 3840)
    sub_fb_shape = (5 * patch_size[0], 10 * patch_size[1])
    f_ratio_h = sub_fb_shape[0] / hf
    f_ratio_w = sub_fb_shape[1] / wf
    vu_points_fisheye = np.vstack((np.round(uv_points[:, 1] / f_ratio_w),
                                   np.round(uv_points[:, 0] / f_ratio_h))).T

    vu_points, uv_points_equi = points_fisheye2equirectangular(vu_points_fisheye, downsampling, camera, rotation_matrix)

    # Replace fisheye in down sampled frame
    uv_points = vu_points[:, ::-1]
    uv_points[:, 0] *= f_ratio_h
    uv_points[:, 1] *= f_ratio_w

    # Replace equirectangular points in sub frame fb_d
    uv_sub_fb_g = np.copy(uv_points_equi)
    uv_sub_fb_g[:, 1] = (uv_sub_fb_g[:, 1] + patch_size[0]) % we_new
    uv_sub_fb_g[:, 0] = (uv_sub_fb_g[:, 0] - patch_size[1]) % he_new

    # Replace equirectangular points in sub frame 1, i.e. fb_d_1
    mask_1 = (uv_sub_fb_g[:, 0] >= 0) & (uv_sub_fb_g[:, 0] < sub_fb_shape[0])
    mask_2 = (uv_sub_fb_g[:, 1] >= 0) & (uv_sub_fb_g[:, 1] < (sub_fb_shape[1] // 2))
    mask = mask_1 & mask_2


    return uv_points[mask, :], uv_sub_fb_g[mask, :]


def fisheye_2_to_fb_g_2(uv_points: np.array,
                        downsampling: int,
                        patch_size: Tuple[int, int],
                        camera: Camera,
                        rotation_matrix: np.array):

    # Default equirectangular specifications
    he, we = (4096, 8192)
    he_new, we_new = he // downsampling, we // downsampling

    # Replace points in original fisheye camera frame 2
    hf, wf = (2160, 3840)
    sub_fb_shape = (5 * patch_size[0], 10 * patch_size[1])
    f_ratio_h = sub_fb_shape[0] / hf
    f_ratio_w = sub_fb_shape[1] / wf
    vu_points_fisheye = np.vstack((np.round((uv_points[:, 1] + sub_fb_shape[1] / 2) / f_ratio_w),
                                   np.round(uv_points[:, 0] / f_ratio_h))).T

    vu_points, uv_points_equi = points_fisheye2equirectangular(vu_points_fisheye, downsampling, camera, rotation_matrix)

    # Replace fisheye in down sampled frame
    uv_points = vu_points[:, ::-1]
    uv_points[:, 0] *= f_ratio_h
    uv_points[:, 1] *= f_ratio_w
    uv_points[:, 1] -= sub_fb_shape[1] // 2

    # Replace equirectangular points in sub frame fb_d
    uv_sub_fb_g = np.copy(uv_points_equi)
    uv_sub_fb_g[:, 1] = (uv_sub_fb_g[:, 1] + patch_size[0]) % we_new
    uv_sub_fb_g[:, 0] = (uv_sub_fb_g[:, 0] - patch_size[1]) % he_new

    # Replace equirectangular points in sub frame 2, i.e. fb_d_2
    mask_1 = (uv_sub_fb_g[:, 0] >= 0) & (uv_sub_fb_g[:, 0] < sub_fb_shape[0])
    mask_2 = (uv_sub_fb_g[:, 1] >= sub_fb_shape[1] // 2) & (uv_sub_fb_g[:, 1] < sub_fb_shape[1])
    mask = mask_1 & mask_2

    uv_sub_fb_g[:, 1] -= sub_fb_shape[1] // 2

    return uv_points[mask, :], uv_sub_fb_g[mask, :]


def fisheye_1_to_fb_a_1(uv_points: np.array,
                        downsampling: int,
                        patch_size: Tuple[int, int],
                        camera: Camera,
                        rotation_matrix: np.array):


    # Default equirectangular specifications
    he, we = (4096, 8192)
    he_new, we_new = he // downsampling, we // downsampling

    # Replace points in original fisheye camera frame 1
    hf, wf = (2160, 3840)
    sub_fb_shape = (5 * patch_size[0], 10 * patch_size[1])
    f_ratio_h = sub_fb_shape[0] / hf
    f_ratio_w = sub_fb_shape[1] / wf
    vu_points_fisheye = np.vstack((np.round(uv_points[:, 1] / f_ratio_w),
                                   np.round(uv_points[:, 0] / f_ratio_h))).T

    vu_points, uv_points_equi = points_fisheye2equirectangular(vu_points_fisheye, downsampling, camera, rotation_matrix)

    # Replace fisheye in down sampled frame
    uv_points = vu_points[:, ::-1]
    uv_points[:, 0] *= f_ratio_h
    uv_points[:, 1] *= f_ratio_w

    # Replace equirectangular points in sub frame fb_d
    uv_sub_fb_a = np.copy(uv_points_equi)
    uv_sub_fb_a[:, 1] = (uv_sub_fb_a[:, 1] - patch_size[0] * 3) % we_new
    msk = uv_sub_fb_a[:, 1] > (we_new // 2)
    uv_sub_fb_a[msk, 1] = (uv_sub_fb_a[msk, 1] - patch_size[0] * 6) % we_new

    # Replace equirectangular points in sub frame 1, i.e. fb_d_1
    mask_1 = (uv_sub_fb_a[:, 0] >= 0) & (uv_sub_fb_a[:, 0] < sub_fb_shape[0])
    mask_2 = (uv_sub_fb_a[:, 1] >= 0) & (uv_sub_fb_a[:, 1] < (sub_fb_shape[1] // 2))
    mask = mask_1 & mask_2


    return uv_points[mask, :], uv_sub_fb_a[mask, :]


def fisheye_2_to_fb_a_2(uv_points: np.array,
                        downsampling: int,
                        patch_size: Tuple[int, int],
                        camera: Camera,
                        rotation_matrix: np.array):

    # Default equirectangular specifications
    he, we = (4096, 8192)
    he_new, we_new = he // downsampling, we // downsampling

    # Replace points in original fisheye camera frame 2
    hf, wf = (2160, 3840)
    sub_fb_shape = (5 * patch_size[0], 10 * patch_size[1])
    f_ratio_h = sub_fb_shape[0] / hf
    f_ratio_w = sub_fb_shape[1] / wf
    vu_points_fisheye = np.vstack((np.round((uv_points[:, 1] + sub_fb_shape[1] / 2) / f_ratio_w),
                                   np.round(uv_points[:, 0] / f_ratio_h))).T

    vu_points, uv_points_equi = points_fisheye2equirectangular(vu_points_fisheye, downsampling, camera, rotation_matrix)

    # Replace fisheye in down sampled frame
    uv_points = vu_points[:, ::-1]
    uv_points[:, 0] *= f_ratio_h
    uv_points[:, 1] *= f_ratio_w
    uv_points[:, 1] -= sub_fb_shape[1] // 2

    # Replace equirectangular points in sub frame fb_d
    uv_sub_fb_a = np.copy(uv_points_equi)
    print(patch_size)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(uv_sub_fb_a[:, 1], bins=64)
    uv_sub_fb_a[:, 1] = (uv_sub_fb_a[:, 1] - patch_size[0] * 3) % we_new
    plt.figure()
    plt.hist(uv_sub_fb_a[:, 1], bins=64)
    msk = uv_sub_fb_a[:, 1] >= (we_new // 2)
    uv_sub_fb_a[msk, 1] = (uv_sub_fb_a[msk, 1] - patch_size[0] * 6) % we_new
    plt.figure()
    plt.hist(uv_sub_fb_a[:, 1], bins=64)
    plt.show()
    # Replace equirectangular points in sub frame 2, i.e. fb_a_2
    mask_1 = (uv_sub_fb_a[:, 0] >= 0) & (uv_sub_fb_a[:, 0] < sub_fb_shape[0])
    mask_2 = (uv_sub_fb_a[:, 1] >= sub_fb_shape[1] // 2) & (uv_sub_fb_a[:, 1] < sub_fb_shape[1])
    mask = mask_1 & mask_2

    uv_sub_fb_a[:, 1] -= sub_fb_shape[1] // 2

    return uv_points[mask, :], uv_sub_fb_a[mask, :]
