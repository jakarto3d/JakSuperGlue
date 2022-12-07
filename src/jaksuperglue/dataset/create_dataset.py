from typing import Tuple
from pathlib import Path
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pyocamcalib.modelling.camera import Camera
import typer
import numpy as np
from loguru import logger
from tqdm import tqdm

from jaksuperglue.dataset.feature_detection import get_sobel_keypoints
from jaksuperglue.utils.dataset_utils import fisheye_1_to_fb_d_1, fisheye_2_to_fb_d_2, fisheye_1_to_fb_g_1, \
    fisheye_2_to_fb_g_2, fisheye_1_to_fb_a_1, fisheye_2_to_fb_a_2, get_sub_fb_d, get_sub_fb_g, get_sub_fb_a, display_lm, \
    write_keypoints

app = typer.Typer()


@app.command()
def create_dataset(root_path: str,
                   calib_d_path: str,
                   calib_g_path: str,
                   calib_a_path: str,
                   downsampling: int = 2,
                   show_lm: bool = False,
                   show_sub_fn: bool = False,
                   output_path: str = None
                   ):
    # Load camera
    camera_g = Camera()
    camera_g.load_parameters(calib_g_path)
    camera_d = Camera()
    camera_d.load_parameters(calib_d_path)
    camera_a = Camera()
    camera_a.load_parameters(calib_a_path)

    # Rotation matrix
    r_g = R.from_euler('yzx', [89.90079489, 1.28823787, -20.58384564], degrees=True).as_matrix()
    r_d = R.from_euler('yzx', [-88.5276023, -3.6436489, -20.81274975], degrees=True).as_matrix()
    r_a = R.from_euler('yzx', [-93.94631351, -47.18336904, -88.81668717], degrees=True).as_matrix()

    if output_path is not None:
        output_path_image = Path(output_path) / 'images'
        output_path_keypoints = Path(output_path) / 'keypoints'

    nb_patch_w = 16
    nb_patch_h = 8
    nb_kpts_patch = 250
    root = Path(root_path)

    sample_dir_list = []
    for template_dir in root.iterdir():
        for sample_dir in template_dir.iterdir():
            if sample_dir.is_dir():
                sample_dir_list.append(sample_dir)

    logger.info(f"Dataset creation parameters: \n"
                f"downsampling = {downsampling} \n"
                f"nb fakebubble = {len(sample_dir_list)} \n"
                f"nb keypoints per image = {nb_kpts_patch}")

    count = 0
    for sample_dir in tqdm(sample_dir_list):
        print(sample_dir.name)
        fb_patch = []
        sphere = cv.imread(str(sample_dir / Path(str(sample_dir.name) + '.jpg')))
        fb = cv.imread(str(sample_dir / Path(str(sample_dir.name) + '.png')))
        for _ in range(downsampling // 2):
            sphere = cv.pyrDown(sphere)
            fb = cv.pyrDown(fb)

        f_d, corr_table_d = camera_d.equirectangular2cam(sphere, np.linalg.inv(r_d), (2160, 3840))
        f_g, corr_table_g = camera_g.equirectangular2cam(sphere, np.linalg.inv(r_g), (2160, 3840))
        f_a, corr_table_a = camera_a.equirectangular2cam(sphere, np.linalg.inv(r_a), (2160, 3840))
        del sphere
        patch_size = (fb.shape[0] // nb_patch_h, fb.shape[1] // nb_patch_w)

        for i in range(nb_patch_h):
            row = []
            for j in range(nb_patch_w):
                row.append(fb[i * patch_size[0]: (i + 1) * patch_size[0],
                           j * patch_size[1]: (j + 1) * patch_size[1]])
            fb_patch.append(row)

        sub_fb_d = get_sub_fb_d(fb_patch, downsampling)
        sub_fb_g = get_sub_fb_g(fb_patch, downsampling)
        sub_fb_a = get_sub_fb_a(fb_patch, downsampling)
        f_d = cv.resize(f_d, (sub_fb_d.shape[1], sub_fb_d.shape[0]))
        f_g = cv.resize(f_g, (sub_fb_g.shape[1], sub_fb_g.shape[0]))
        f_a = cv.resize(f_a, (sub_fb_a.shape[1], sub_fb_a.shape[0]))


        for counter, (img_1, img_2) in enumerate(zip([sub_fb_d, sub_fb_g, sub_fb_a], [f_d, f_g, f_a])):

            # Crop image in two equal parts
            im_size = (img_1.shape[0], sub_fb_d.shape[1] // 2)
            im_1_pc = img_1[:, :im_size[1]]
            im_1_rgb = img_2[:, :im_size[1]]
            im_2_pc = img_1[:, im_size[1]:]
            im_2_rgb = img_2[:, im_size[1]:]

            # Extract keypoints
            best_uv_point_1 = get_sobel_keypoints(im_1_rgb, nb_kpts_patch)
            best_uv_point_2 = get_sobel_keypoints(im_2_rgb, nb_kpts_patch)


            if counter == 0:
                # Get corresponding points in sub_fb_d
                best_uv_point_1, best_uv_point_pc_1 = fisheye_1_to_fb_d_1(best_uv_point_1,
                                                                          downsampling,
                                                                          patch_size,
                                                                          camera_d,
                                                                          np.linalg.inv(r_d))

                best_uv_point_2, best_uv_point_pc_2 = fisheye_2_to_fb_d_2(best_uv_point_2,
                                                                          downsampling,
                                                                          patch_size,
                                                                          camera_d,
                                                                          np.linalg.inv(r_d))

            elif counter == 1:
                # Get corresponding points in sub_fb_g
                best_uv_point_1, best_uv_point_pc_1 = fisheye_1_to_fb_g_1(best_uv_point_1,
                                                                          downsampling,
                                                                          patch_size,
                                                                          camera_g,
                                                                          np.linalg.inv(r_g))

                best_uv_point_2, best_uv_point_pc_2 = fisheye_2_to_fb_g_2(best_uv_point_2,
                                                                          downsampling,
                                                                          patch_size,
                                                                          camera_g,
                                                                          np.linalg.inv(r_g))

            else:
                # Get corresponding points in sub_fb_a
                best_uv_point_1, best_uv_point_pc_1 = fisheye_1_to_fb_a_1(best_uv_point_1,
                                                                          downsampling,
                                                                          patch_size,
                                                                          camera_a,
                                                                          np.linalg.inv(r_a))

                best_uv_point_2, best_uv_point_pc_2 = fisheye_2_to_fb_a_2(best_uv_point_2,
                                                                          downsampling,
                                                                          patch_size,
                                                                          camera_a,
                                                                          np.linalg.inv(r_a))
            if output_path is not None:
                if counter != 2:
                    cv.imwrite(str(output_path_image / f'{count}_pc.png'), im_1_pc)
                    cv.imwrite(str(output_path_image / f'{count}_fi.png'), im_1_rgb)
                    write_keypoints(best_uv_point_1, best_uv_point_pc_1, output_path_keypoints, count)
                    count += 1

                cv.imwrite(str(output_path_image / f'{count}_pc.png'), im_2_pc)
                cv.imwrite(str(output_path_image / f'{count}_fi.png'), im_2_rgb)
                write_keypoints(best_uv_point_2, best_uv_point_pc_2, output_path_keypoints, count)
                count += 1

            if show_lm:
                display_lm(im_1_rgb, im_1_pc, best_uv_point_1, best_uv_point_pc_1, name=None)
                display_lm(im_2_rgb, im_2_pc, best_uv_point_2, best_uv_point_pc_2, name=None)

        if show_sub_fn:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(f_d[:, :, ::-1])
            plt.title('Fisheye d', fontsize=20)
            plt.subplot(1, 2, 2)
            plt.imshow(sub_fb_d[:, :, ::-1])
            plt.title('Sub fakebubble d', fontsize=20)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(f_g[:, :, ::-1])
            plt.title('Fisheye g', fontsize=20)
            plt.subplot(1, 2, 2)
            plt.imshow(sub_fb_g[:, :, ::-1])
            plt.title('Sub fakebubble g', fontsize=20)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(f_a[:, :, ::-1])
            plt.title('Fisheye a', fontsize=20)
            plt.subplot(1, 2, 2)
            plt.imshow(sub_fb_a[:, :, ::-1])
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.title('Sub fakebubble a', fontsize=20)

            plt.show()


def get_fake_fisheye(sphere: np.array,
                     camera: Camera,
                     rotation_matrix: np.array,
                     output_size: Tuple[int, int]):
    fake_fisheye = camera.equirectangular2cam(sphere, rotation_matrix, output_size)
    return fake_fisheye


def load_image(image_path: str, mask: np.array):
    img = cv.imread(image_path)
    if mask is not None:
        img[mask] = 0
    return img


if __name__ == '__main__':
    app()
