import errno
from typing import Tuple, Union
import json
from pathlib import Path
import numpy as np
import os

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def write_keypoints(fi_kpts: np.array,
                    pc_kpts: np.array,
                    conf: np.array,
                    im_type: str,
                    sphere_name: str,
                    output_path: str):
    output_path = Path(output_path)

    fi_kpts = fi_kpts
    pc_kpts = pc_kpts

    res = {f'vu_fi': fi_kpts.tolist(),
           f'vu_subfb': pc_kpts.tolist(),
           'confidence': conf.tolist()
          }

    with open(output_path / f'{sphere_name}_{im_type}.json', 'w') as f:
        json.dump(res, f)

def save_preprocessing_meta(output_path: Union[str, Path],
                            metadata: dict):

    with open(str(output_path / 'preprocessing_meta.json'), 'w') as f:
        json.dump(metadata, f)

def load_preprocessing_meta(file_path: Union[str, Path]):
    with open(str(file_path), 'r') as f:
        meta = json.load(f)
    return meta

def load_keypoints(file_path: Union[Path, str]):
    with open(str(file_path), 'r') as f:
        data = json.load(f)

    kpts_fi = np.array(data['vu_fi'])
    kpts_subfb = np.array(data['vu_subfb'])
    conf = np.array(data['confidence'])

    return kpts_fi, kpts_subfb, conf


def rotate_kpts(kpts, im_size: Tuple[int, int], angle: float):
    # center kpts before rotation
    kpts[:, 0] -= im_size[1] / 2
    kpts[:, 1] -= im_size[0] / 2

    angle_rad = np.deg2rad(angle)
    M = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_kpts = kpts @ M.T

    rotated_kpts[:, 0] += im_size[1] / 2
    rotated_kpts[:, 1] += im_size[0] / 2

    return rotated_kpts


def back2original_fi_1(kpts: np.array,
                       meta: dict,
                       rotation_angle: float = None):

    if rotation_angle is not None:
        kpts = rotate_kpts(kpts, meta['c_fi_size'], rotation_angle)

    ratio_h = meta['o_fi_size'][0] / meta['c_fi_size'][0]
    ratio_w = meta['o_fi_size'][1] / meta['c_fi_size'][1]
    kpts[:, 0] *= ratio_w
    kpts[:, 1] *= ratio_h

    return kpts


def back2original_fi_2(kpts: np.array,
                       meta: dict,
                       rotation_angle: float = None):

    if rotation_angle is not None:
        im_size = (meta['c_fi_size'][0],  meta['c_fi_size'][1] // 2)
        kpts = rotate_kpts(kpts, im_size, rotation_angle)

    ratio_h = meta['o_fi_size'][0] / meta['c_fi_size'][0]
    ratio_w = meta['o_fi_size'][1] / meta['c_fi_size'][1]
    middle = meta['c_fi_size'][1] // 2
    kpts[:, 0] += middle
    kpts[:, 0] *= ratio_w
    kpts[:, 1] *= ratio_h

    return kpts


def back2original_subfb_d1(kpts: np.array,
                           meta: dict):


    we_resize = meta['o_fb_size'][1] / meta['downsampling']
    he_resize = meta['o_fb_size'][0] / meta['downsampling']

    # Replace points in subfb_d in equirectangular frame
    kpts[:, 0] = (kpts[:, 0] + meta['fragment_size'][1] * 7) % we_resize
    kpts[:, 1] = (kpts[:, 1] + meta['fragment_size'][0]) % he_resize

    kpts[:, 0] *= meta['downsampling']
    kpts[:, 1] *= meta['downsampling']

    return kpts


def back2original_subfb_d2(kpts: np.array,
                           meta: dict):

    we_resize = meta['o_fb_size'][1] / meta['downsampling']
    he_resize = meta['o_fb_size'][0] / meta['downsampling']

    kpts[:, 0] += meta['c_subfb_size'][1] / 2

    # Replace points in subfb_d in equirectangular frame
    kpts[:, 0] = (kpts[:, 0] + meta['fragment_size'][0] * 7) % we_resize
    kpts[:, 1] = (kpts[:, 1] + meta['fragment_size'][1]) % he_resize

    kpts[:, 0] *= meta['downsampling']
    kpts[:, 1] *= meta['downsampling']

    return kpts


def back2original_subfb_g1(kpts: np.array,
                           meta: dict):

    we_resize = meta['o_fb_size'][1] / meta['downsampling']
    he_resize = meta['o_fb_size'][0] / meta['downsampling']

    # Replace points in subfb_d in equirectangular frame
    kpts[:, 0] = (kpts[:, 0] - meta['fragment_size'][1]) % we_resize
    kpts[:, 1] = (kpts[:, 1] + meta['fragment_size'][0]) % he_resize

    kpts[:, 0] *= meta['downsampling']
    kpts[:, 1] *= meta['downsampling']

    return kpts


def back2original_subfb_g2(kpts: np.array,
                           meta: dict):

    we_resize = meta['o_fb_size'][1] / meta['downsampling']
    he_resize = meta['o_fb_size'][0] / meta['downsampling']

    kpts[:, 0] += meta['c_subfb_size'][1] / 2

    # Replace points in subfb_d in equirectangular frame
    kpts[:, 0] = (kpts[:, 0] - meta['fragment_size'][1]) % we_resize
    kpts[:, 1] = (kpts[:, 1] + meta['fragment_size'][0]) % he_resize

    kpts[:, 0] *= meta['downsampling']
    kpts[:, 1] *= meta['downsampling']

    return kpts


def back2original_subfb_a2(kpts: np.array,
                           meta: dict):

    we_resize = meta['o_fb_size'][1] / meta['downsampling']
    he_resize = meta['o_fb_size'][0] / meta['downsampling']

    kpts[:, 0] += meta['c_subfb_size'][1] / 2


    # Replace points in subfb_d in equirectangular frame

    kpts[:, 0] = (kpts[:, 0] + meta['fragment_size'][1] * 9) % we_resize

    kpts[:, 0] *= meta['downsampling']
    kpts[:, 1] *= meta['downsampling']

    return kpts
