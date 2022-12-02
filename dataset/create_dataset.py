from typing import Tuple, List
from pathlib import Path
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from loguru import logger
from pyocamcalib.modelling.camera import Camera
import typer
import numpy as np

app = typer.Typer()

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
        for j, col_id in enumerate([11, 12, 13, 14, 15, 0, 1, 2, 3, 4]):
            sub_sphere_a[i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]] = \
                patch_list[row_id][col_id]

    return sub_sphere_a

def get_corresponding_pc_points(uv_kpts_rgb: np.array,
                                dict_fisheye: np.array,
                                dict_fb: np.array,
                                patch_size: Tuple[int, int]):
    kpts_pc = []
    for kpt in uv_kpts_rgb:
        index = np.where((dict_fisheye[:, 0] == kpt[0]) & (dict_fisheye[:, 1] == kpt[1]))[0][0]
        kpts_pc.append(dict_fb[index])

    kpts_pc = np.array(kpts_pc)
    mask_1 = (kpts_pc[:, 0] >= 0) & (kpts_pc[:, 0] < patch_size[0])
    mask_2 = (kpts_pc[:, 1] >= 0) & (kpts_pc[:, 1] < patch_size[1])
    mask = mask_1 & mask_2
    return uv_kpts_rgb[mask], kpts_pc[mask]

@app.command()
def create_dataset(root_path: str,
                   calib_d_path: str,
                   calib_g_path: str,
                   calib_a_path: str,
                   downsampling: int = 2,
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

    fb_patch = []
    f_d_patch = []
    f_g_patch = []
    f_a_patch = []

    nb_patch_w = 16
    nb_patch_h = 8
    nb_kpts_patch = 150
    root = Path(root_path)
    import random
    random.seed(11)
    for template_dir in random.sample(list(root.iterdir()), 1):
        for sample_dir in template_dir.iterdir():
            fb_patch = []
            f_d_patch = []
            f_g_patch = []
            f_a_patch = []
            if sample_dir.is_dir():
                sphere = cv.imread(str(sample_dir / Path(str(sample_dir.name) + '.jpg')))
                fb = cv.imread(str(sample_dir / Path(str(sample_dir.name) + '.png')))
                sphere_down = cv.pyrDown(sphere,
                                         dstsize=(sphere.shape[1] // downsampling, sphere.shape[0] // downsampling))
                fb_down = cv.pyrDown(fb, dstsize=(fb.shape[1] // downsampling, fb.shape[0] // downsampling))
                f_d, corr_table_d = camera_d.equirectangular2cam(sphere_down, np.linalg.inv(r_d), (2160, 3840))
                f_g, corr_table_g = camera_g.equirectangular2cam(sphere_down, np.linalg.inv(r_g), (2160, 3840))
                f_a, corr_table_a = camera_a.equirectangular2cam(sphere_down, np.linalg.inv(r_a), (2160, 3840))
                del sphere, sphere_down
                del fb
                patch_size = (fb_down.shape[0] // nb_patch_h, fb_down.shape[1] // nb_patch_w)

                for i in range(nb_patch_h):
                    row = []
                    for j in range(nb_patch_w):
                        row.append(fb_down[i * patch_size[0]: (i + 1) * patch_size[0],
                                   j * patch_size[1]: (j + 1) * patch_size[1]])
                    fb_patch.append(row)

                sub_fb_d = get_sub_fb_d(fb_patch, downsampling)
                sub_fb_g = get_sub_fb_g(fb_patch, downsampling)
                sub_fb_a = get_sub_fb_a(fb_patch, downsampling)
                f_d = cv.resize(f_d, (sub_fb_d.shape[1], sub_fb_d.shape[0]))
                f_g = cv.resize(f_g, (sub_fb_g.shape[1], sub_fb_g.shape[0]))
                f_a = cv.resize(f_a, (sub_fb_a.shape[1], sub_fb_a.shape[0]))
                f_ratio_h = sub_fb_d.shape[0] / 2160
                f_ratio_w = sub_fb_d.shape[1] / 3840

                uv_points_fisheye = np.vstack((np.round(corr_table_d[:, 1] * f_ratio_h),
                                               np.round(corr_table_d[:, 0] * f_ratio_w))).T


                uv_sub_fb_d = np.vstack((np.round(corr_table_d[:, 2]),
                                         np.round(corr_table_d[:, 3]))).T
                uv_sub_fb_d[:, 1] = (uv_sub_fb_d[:, 1] - patch_size[0] * 7) % fb_down.shape[1]
                uv_sub_fb_d[:, 0] = (uv_sub_fb_d[:, 0] - patch_size[1]) % fb_down.shape[0]

                uv_sub_fb_g = np.vstack((np.round(corr_table_g[:, 2] / downsampling),
                                         np.round(corr_table_g[:, 2] / downsampling))).T
                uv_sub_fb_g[:, 1] = (uv_sub_fb_g[:, 1] - patch_size[0] * 15) % fb_down.shape[1]
                uv_sub_fb_g[:, 0] = (uv_sub_fb_g[:, 0] - patch_size[1]) % fb_down.shape[0]

                uv_sub_fb_a = np.vstack((np.round(corr_table_a[:, 2] / downsampling),
                                         np.round(corr_table_a[:, 2] / downsampling))).T
                uv_sub_fb_a[:, 1] = (uv_sub_fb_a[:, 1] - patch_size[0] * 10) % fb_down.shape[1]

                sift = cv.SIFT_create()

                for img_1, img_2 in zip([sub_fb_d, sub_fb_g, sub_fb_a], [f_d, f_g, f_a]):
                    im_size = (img_1.shape[0], sub_fb_d.shape[1] // 2)
                    im_1_pc = img_1[:, :im_size[1]]
                    im_1_rgb = img_2[:, :im_size[1]]
                    im_2_pc = img_1[:, im_size[1]:]
                    im_2_rgb = img_2[:, im_size[1]:]
                    im_1_gray = cv.cvtColor(im_1_rgb, cv.COLOR_BGR2GRAY)
                    im_2_gray = cv.cvtColor(im_2_rgb, cv.COLOR_BGR2GRAY)
                    kp_im_1, des1 = sift.detectAndCompute(im_1_gray, None)
                    kp_im_2, des1 = sift.detectAndCompute(im_2_gray, None)
                    u_1 = [kp.pt[1] for kp in kp_im_1]
                    v_1 = [kp.pt[0] for kp in kp_im_1]
                    uv_point_1 = np.vstack((u_1, v_1)).T
                    score_1 = [kp.response for kp in kp_im_1]
                    size_1 = np.array([kp.size for kp in kp_im_1])
                    index_size_1 = np.where(size_1 > 4)[0]
                    index_best_1 = np.argsort(np.array(score_1)[index_size_1])[::-1][:nb_kpts_patch]
                    best_uv_point_1 = np.round(uv_point_1[index_size_1][index_best_1])

                    best_uv_point_1, best_uv_point_pc_1 = get_corresponding_pc_points(best_uv_point_1,
                                                                                      uv_points_fisheye,
                                                                                      uv_sub_fb_d, im_size)

                    u_2 = [kp.pt[1] for kp in kp_im_2]
                    v_2 = [kp.pt[0] for kp in kp_im_2]
                    uv_point_2 = np.vstack((u_2, v_2)).T
                    score_2 = [kp.response for kp in kp_im_2]
                    size_2 = np.array([kp.size for kp in kp_im_2])
                    index_size_2 = np.where(size_2 > 4)[0]
                    index_best_2 = np.argsort(np.array(score_2)[index_size_2])[::-1][:nb_kpts_patch]
                    best_uv_point_2 = uv_point_2[index_size_2][index_best_2]

                    display_lm(im_1_rgb[:, :, ::-1], im_1_pc, best_uv_point_1, best_uv_point_pc_1, name=None)
                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(im_1_rgb[:, :, ::-1])
                    plt.scatter(best_uv_point_1[:, 1], best_uv_point_1[:, 0], marker='+', c='r', s=40)
                    plt.subplot(1, 2, 2)
                    plt.imshow(im_2_rgb[:, :, ::-1])
                    plt.scatter(best_uv_point_2[:, 1], best_uv_point_2[:, 0], marker='+', c='r', s=40)
                    plt.show()

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
    # from pyocamcalib.modelling.camera import Camera
    # from pathlib import Path
    # import cv2 as cv
    # import matplotlib
    # import matplotlib.pyplot as plt
    # from scipy.spatial.transform import Rotation as R
    #
    # matplotlib.use("TkAgg")
    #
    # calib_left_path = "/home/hugo/Jakarto/Jaksphere/data/Calib/calibration_fisheye_left_16052022_175041.json"
    # calib_right_path = "/home/hugo/Jakarto/Jaksphere/data/Calib/calibration_fisheye_right_17052022_094555.json"
    # calib_back_path = "/home/hugo/Jakarto/Jaksphere/data/Calib/calibration_fisheye_back_17052022_095123.json"
    #
    # img_file_a = []
    # img_file_g = []
    # img_file_d = []
    # img_file_ad = []
    # img_file_ag = []
    #
    # working_dir = Path('/home/hugo/Jakarto/Jaksphere/data/jak_raw')
    # img_file = sorted(list(working_dir.glob('**/*.webp')))
    #
    # for e in img_file:
    #     if e.name == 'a.webp':
    #         img_file_a.append(e)
    #     elif e.name == 'g.webp':
    #         img_file_g.append(e)
    #     elif e.name == 'd.webp':
    #         img_file_d.append(e)
    #     elif e.name == 'ad.webp':
    #         img_file_ad.append(e)
    #     elif e.name == 'ag.webp':
    #         img_file_ag.append(e)
    #
    # mask_file_a = '/home/hugo/Jakarto/Jaksphere/data/jak_raw/masks/mask_a.png'
    # mask_file_g = '/home/hugo/Jakarto/Jaksphere/data/jak_raw/masks/mask_g.png'
    # mask_file_d = '/home/hugo/Jakarto/Jaksphere/data/jak_raw/masks/mask_d.png'
    #
    # msk_a = cv.imread(mask_file_a) == 0
    # msk_g = cv.imread(mask_file_g) == 0
    # msk_d = cv.imread(mask_file_d) == 0
    #
    # fisheye_a = load_image(str(img_file_a[0]), msk_a)
    # fisheye_d = load_image(str(img_file_d[0]), msk_d)
    # fisheye_g = load_image(str(img_file_g[0]), msk_g)
    #
    # my_camera_g = Camera()
    # my_camera_g.load_parameters(calib_left_path)
    # my_camera_d = Camera()
    # my_camera_d.load_parameters(calib_right_path)
    # my_camera_a = Camera()
    # my_camera_a.load_parameters(calib_back_path)
    #
    # # plt.figure("fisheye")
    # # plt.imshow(fisheye_d[:, :, ::-1])
    # # plt.figure("perspective projection direct")
    # # plt.imshow(my_camera_d.cam2perspective_direct(fisheye_d, fov=120, sensor_size=(700, 700))[:, :, ::-1])
    # # plt.figure("perspective projection indirect ")
    # # plt.imshow(my_camera_d.cam2perspective_direct(fisheye_d, fov=120, sensor_size=(1400, 1400))[:, :, ::-1])
    # # plt.show()
    # equi_path = "/media/hugo/T7/hugo/Jakarto3D_KeypointsMatching/template_155/jak1_20211019_77774s902ms/jak1_20211019_77774s902ms.jpg"
    # r = R.from_euler('yzx', [-88.5276023, -3.6436489, -20.81274975], degrees=True).as_matrix()
    # equi = cv.imread(equi_path)
    # reconstructed_fisheye = my_camera_d.equirectangular2cam(equi, np.linalg.inv(r), (2160, 3840))
    #
    # plt.figure()
    # plt.imshow(reconstructed_fisheye[:, :, ::-1])
    # plt.show()
