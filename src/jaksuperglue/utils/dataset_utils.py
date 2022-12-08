from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
import json
import cv2 as cv
from pyocamcalib.modelling.camera import Camera, cartesian2geographic
import numpy as np


def get_fakebubble_fragment(fakebubble_downsample: np.array,
                            nb_fragment_h: int,
                            nb_fragment_w: int):
    fragments_list = []
    fragment_size = (fakebubble_downsample.shape[0] // nb_fragment_h,
                     fakebubble_downsample.shape[1] // nb_fragment_w)

    for i in range(nb_fragment_h):
        row = []
        for j in range(nb_fragment_w):
            fragment = fakebubble_downsample[i * fragment_size[0]: (i + 1) * fragment_size[0],
                       j * fragment_size[1]: (j + 1) * fragment_size[1]]
            row.append(fragment)
        fragments_list.append(row)

    return fragments_list, fragment_size


def load_sphere(sphere_path: str,
                downsampling: int):
    """
    sphere_path: path of the sphere to load
    downsampling: should be a multiple of 2
    """
    sphere = cv.imread(sphere_path, 0)
    for _ in range(downsampling // 2):
        sphere = cv.pyrDown(sphere)

    return sphere


def load_fisheye(fisheye_path: str,
                 resize: Tuple[int, int]):
    f_im = cv.imread(fisheye_path, 0)
    f_im = cv.resize(f_im, (resize[1], resize[0]))

    return f_im


def split_image_in_two(im: np.array):
    middle = im.shape[1] // 2
    im_1 = im[:, :middle]
    im_2 = im[:, middle:]
    return im_1, im_2

def display_lm(img_target, img_src, uv_target, uv_src, name=None):
    H_t, W_t = img_target.shape[0], img_target.shape[1]
    H_s, W_s = img_src.shape[0], img_src.shape[1]
    if (img_target.ndim > 2) and (img_src.ndim > 2):
        rgb_flag = True
        background = img_target[:, :, [2, 1, 0]].copy()
    else:
        rgb_flag = False
        background = img_target.copy()

    shift_h = int(np.round(np.abs(H_t - H_s) / 2) * (H_s - H_t > 0)) + 1
    if rgb_flag:
        pad_width = [(shift_h, shift_h), (0, W_s), (0, 0)]
    else:
        pad_width = [(shift_h, shift_h), (0, W_s)]
    background = np.pad(background, pad_width)

    if rgb_flag:
        background[:H_s, W_t:, :] = img_src[:H_s, :, [2, 1, 0]].copy()
    else:
        background[:H_s, W_t:] = img_src[:H_s, :].copy()

    fig, ax = plt.subplots(ncols=1, figsize=(20, 20))
    if rgb_flag:
        ax.imshow(background)
    else:
        ax.imshow(background, cmap='gray')

    for i in range(uv_target.shape[0]):
        ax.scatter(uv_target[i, 1], uv_target[i, 0] + shift_h,
                   s=20, c='g', marker='+')
        ax.scatter(uv_src[i, 1] + W_t, uv_src[i, 0], s=20, c='g', marker='+')
        ax.plot([uv_target[i, 1], uv_src[i, 1] + W_t],
                [uv_target[i, 0] + shift_h, uv_src[i, 0]], c='g',
                markersize=20)
    if name is not None:
        plt.savefig(name, dpi=300)
    else:
        plt.show()

def get_sub_fb_d(patch_list: List[np.array],
                 downsampling: int):
    patch_size = patch_list[0][0].shape[:2]
    sub_sphere_d = np.zeros((2560 // downsampling, 5120 // downsampling, 3)).astype(np.uint8)
    for i, row_id in enumerate([1, 2, 3, 4, 5]):
        for j, col_id in enumerate([7, 8, 9, 10, 11, 12, 13, 14, 15, 0]):
            sub_sphere_d[i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]] = \
                patch_list[row_id][col_id]

    return sub_sphere_d


def get_sub_fb_g(patch_list: List[np.array],
                 downsampling: int):
    patch_size = patch_list[0][0].shape[:2]
    sub_sphere_g = np.zeros((2560 // downsampling, 5120 // downsampling, 3)).astype(np.uint8)
    for i, row_id in enumerate([1, 2, 3, 4, 5]):
        for j, col_id in enumerate([15, 0, 1, 2, 3, 4, 5, 6, 7, 8]):
            sub_sphere_g[i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]] = \
                patch_list[row_id][col_id]

    return sub_sphere_g


def get_sub_fb_a(patch_list: List[np.array],
                 downsampling: int):
    patch_size = patch_list[0][0].shape[:2]
    sub_sphere_a = np.zeros((2560 // downsampling, 5120 // downsampling, 3)).astype(np.uint8)
    for i, row_id in enumerate([0, 1, 2, 3, 4]):
        for j, col_id in enumerate([3, 4, 11, 12, 13,14, 15, 0, 1, 2]):
            sub_sphere_a[i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]] = \
                patch_list[row_id][col_id]

    return sub_sphere_a


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
    uv_points = np.round(uv_points).astype(int)

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
    uv_points[:, 1] -= sub_fb_shape[1] / 2
    uv_points = np.round(uv_points).astype(int)

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
    uv_points = np.round(uv_points).astype(int)

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
    uv_points[:, 1] -= sub_fb_shape[1] / 2
    uv_points = np.round(uv_points).astype(int)

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
    uv_points = np.round(uv_points).astype(int)

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
    uv_points[:, 1] -= sub_fb_shape[1] / 2
    uv_points = np.round(uv_points).astype(int)

    # Replace equirectangular points in sub frame fb_a
    uv_sub_fb_a = np.copy(uv_points_equi)
    uv_sub_fb_a[:, 1] = (uv_sub_fb_a[:, 1] - patch_size[0] * 3) % we_new
    msk = uv_sub_fb_a[:, 1] >= (we_new // 2)
    uv_sub_fb_a[msk, 1] = (uv_sub_fb_a[msk, 1] - patch_size[0] * 6) % we_new

    # Replace equirectangular points in sub frame 2, i.e. fb_a_2
    mask_1 = (uv_sub_fb_a[:, 0] >= 0) & (uv_sub_fb_a[:, 0] < sub_fb_shape[0])
    mask_2 = (uv_sub_fb_a[:, 1] >= sub_fb_shape[1] // 2) & (uv_sub_fb_a[:, 1] < sub_fb_shape[1])
    mask = mask_1 & mask_2

    uv_sub_fb_a[:, 1] -= sub_fb_shape[1] // 2

    return uv_points[mask, :], uv_sub_fb_a[mask, :]

def write_keypoints(kpts_1: np.array,  kpts_2: np.array, output_file: Path, value: int):
    kpts = {'fisheye': kpts_1.tolist(),
            'pc': kpts_2.tolist()}
    with open(output_file / f'kpts_{value}.json', 'w') as f:
        json.dump(kpts, f)
