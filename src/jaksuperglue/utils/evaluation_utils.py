import numpy as np
from pyocamcalib.modelling.camera import Camera
from jaksuperglue.utils.dataset_utils import fisheye_1_to_fb_d_1, fisheye_2_to_fb_d_2, fisheye_1_to_fb_g_1, \
    fisheye_2_to_fb_g_2, fisheye_2_to_fb_a_2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_score(mkpts0: np.array,
              mkpts1: np.array,
              m_conf: np.array,
              im_type: str,
              dataset_config: dict,
              thresh: float,
              im):
    msk = m_conf > thresh

    valid_mkpts0 = mkpts0[msk]
    valid_mkpts1 = mkpts1[msk]
    valid_mkpts0 = valid_mkpts0[:, ::-1]
    valid_mkpts1 = valid_mkpts1[:, ::-1]

    camera = Camera()

    if im_type == 'd1' or im_type == 'd2':
        camera.load_parameters(dataset_config['calib_d_path'])
        rotation_matrix = dataset_config['rotation_d']
    elif im_type == 'g1' or im_type == 'g2':
        camera.load_parameters(dataset_config['calib_g_path'])
        rotation_matrix = dataset_config['rotation_g']
    elif im_type == 'a1' or im_type == 'a2':
        camera.load_parameters(dataset_config['calib_a_path'])
        rotation_matrix = dataset_config['rotation_a']
    else:
        raise ValueError(f'wrong im_type: {im_type}')

    projection_function = type2projection(im_type)
    new_mkpts0, reprojected_mkpts0 = projection_function(valid_mkpts0,
                                                         dataset_config['downsampling'],
                                                         dataset_config['patch_size'],
                                                         camera,
                                                         np.linalg.inv(rotation_matrix))


    new_mkpts0 = new_mkpts0.astype(np.float32)
    A = [str(e) for e in new_mkpts0]
    B = [str(e) for e in valid_mkpts0]
    C, x_ind, y_ind = np.intersect1d(A, B, return_indices=True)

    nb_kpts = reprojected_mkpts0.shape[0]
    if nb_kpts > 0:
        dist = np.sqrt(
            (reprojected_mkpts0[x_ind, 0] - valid_mkpts1[y_ind, 0]) ** 2 + (
                        reprojected_mkpts0[x_ind, 1] - valid_mkpts1[y_ind, 1]) ** 2)

        # print(f'{dist=}')
        # print(f'{np.mean(dist)=}')
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(im, cmap='gray')
        # plt.scatter(reprojected_mkpts0[x_ind, 1], reprojected_mkpts0[x_ind, 0], marker='+', c='g', s=40)
        # plt.scatter(valid_mkpts1[y_ind, 1], valid_mkpts1[y_ind, 0], marker='+', c='r', s=40)
        # plt.show()
        return np.mean(dist), np.std(dist), nb_kpts
    else:
        return np.nan, np.nan, nb_kpts


def type2int(im_type: str):
    if im_type == 'd1':
        return 0
    elif im_type == 'd2':
        return 1
    elif im_type == 'g1':
        return 2
    elif im_type == 'g2':
        return 3
    elif im_type == 'a2':
        return 4


def type2projection(im_type: str):
    if im_type == 'd1':
        return fisheye_1_to_fb_d_1
    elif im_type == 'd2':
        return fisheye_2_to_fb_d_2
    elif im_type == 'g1':
        return fisheye_1_to_fb_g_1
    elif im_type == 'g2':
        return fisheye_2_to_fb_g_2
    elif im_type == 'a2':
        return fisheye_2_to_fb_a_2


def show_evaluation(df: pd.DataFrame):
    plt.figure(figsize=(10, 10))
    sns.lineplot(x="threshold", y="nb_kpts",
                 hue="im_type", data=df).set(title='number of match function of confidence')

    plt.figure(figsize=(10, 10))
    sns.lineplot(x="threshold", y="precision",
                 hue="im_type", data=df).set(title='number of matchs function of confidence threshold',
                                                         xlabel="threshold (good match probability)",
                                                         ylabel="precision (distance in pixel)", )



