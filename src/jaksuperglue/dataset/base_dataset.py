from pathlib import Path
from typing import Union, Literal
import json
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

def get_folders(root: Union[str, Path]):
    sample_dir_list = []
    for sample_dir in root.iterdir():
        if sample_dir.is_dir():
            sample_dir_list.append(sample_dir)

    return sample_dir_list


def check_sample_id(file_1: Path, file_2: Path):
    return file_1.name.split('_')[0] == file_2.name.split('_')[0]

def get_type(im_file: Path):
    return im_file.name.split('_')[-1].split('.')[0]

def load_keypoints(kpts_file: Path):
    with open(str(kpts_file), 'r') as f:
        kpts = json.load(f)

    fi_kpts = np.array(kpts['fisheye'])
    pc_kpts = np.array(kpts['pc'])

    return fi_kpts, pc_kpts


class JakOnlyImageDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self,
                 root: Union[str, Path],
                 mode: Literal["train", "eval"]):

        self.root = Path(root)
        self.data_path = self.root / mode
        self.image_path = self.data_path / 'images'
        self.keypoints_path = self.data_path / 'keypoints'
        self.mode = mode
        self.images_fi_list = list(self.image_path.glob("*_fi_*.png"))
        self.images_pc_list = list(self.image_path.glob("*_pc_*.png"))
        self.kpts_list = list(self.keypoints_path.glob("*.json"))

    def __len__(self):
        return len(self.kpts_list)

    def __getitem__(self, idx):
        kpts_file = self.kpts_list[idx]
        fi_file = self.images_fi_list[idx]
        pc_file = self.images_pc_list[idx]

        assert check_sample_id(kpts_file, fi_file) and check_sample_id(fi_file, pc_file)

        fi_im = cv.imread(str(fi_file), 0)
        pc_im = cv.imread(str(pc_file), 0)

        fi_kpts, pc_kpts = load_keypoints(kpts_file)

        return {
            'fast_keypoints0': fi_kpts,
            'fast_keypoints1': pc_kpts,
            'image0': fi_im,
            'image1': pc_im,
            'type':get_type(fi_file),
            'sample_id': kpts_file.name.split('_')[0]
        }

class JakOnlyImageCollator(object):
    def __init__(self, device=torch.device("cuda")):
        self.device = device

    def __call__(self, inputs):
        inputs = inputs[0]
        inputs_torch = {'fast_keypoints0': inputs['fast_keypoints0'].reshape((1, -1, 2)),
                        'fast_keypoints1': inputs['fast_keypoints1'].reshape((1, -1, 2)),
                        'image0':  torch.from_numpy(inputs['image0'] / 255.).float()[None, None].to(self.device),
                        'image1':  torch.from_numpy(inputs['image1'] / 255.).float()[None, None].to(self.device),
                        'type': inputs['type'],
                        'sample_id': inputs['sample_id']}

        return inputs_torch