from pathlib import Path
from typing import Union, Tuple
import cv2 as cv
import json
from tqdm import tqdm
from loguru import logger

from jaksuperglue.utils.dataset_utils import load_sphere, load_fisheye, get_sub_fb_d, get_sub_fb_g, get_sub_fb_a, \
    get_fakebubble_fragment, split_image_in_two
from jaksuperglue.utils.inference_utils import save_preprocessing_meta, mkdir


class PreProcessingEngine:
    def __init__(self,
                 working_dir: Union[Path, str],
                 downsampling: int = 2,
                 nb_fragment: Tuple[int, int] = (8, 16)):
        self.working_dir = Path(working_dir)
        self.out_preprocessing = Path(self.working_dir.parents[0] / 'preprocessing')
        self.out_images = Path(self.out_preprocessing / 'to_process')
        self.downsampling = downsampling
        self.nb_fragment = nb_fragment

    def process(self):
        metadata = {}
        logger.info('Image Matching Preprocessing: Start')
        mkdir(str(self.out_preprocessing))
        mkdir(str(self.out_images))

        for directory in tqdm(list(self.working_dir.iterdir())):
            if directory.is_dir():
                fb_path = str(directory / Path(str(directory.name) + '.png'))
                f_d_path = str(directory / 'd.tiff')
                f_g_path = str(directory / 'g.tiff')
                f_a_path = str(directory / 'a.tiff')
                fb = load_sphere(fb_path, self.downsampling)

                fakebubble_fragments, fragment_size = get_fakebubble_fragment(fb, self.nb_fragment[0], self.nb_fragment[1])

                sub_fb_d = get_sub_fb_d(fakebubble_fragments, self.downsampling)
                sub_fb_g = get_sub_fb_g(fakebubble_fragments, self.downsampling)
                sub_fb_a = get_sub_fb_a(fakebubble_fragments, self.downsampling)

                f_d, o_fi_size = load_fisheye(f_d_path, resize=sub_fb_d.shape[:2])
                f_g, o_fi_size = load_fisheye(f_g_path, resize=sub_fb_g.shape[:2])
                f_a, o_fi_size = load_fisheye(f_a_path, resize=sub_fb_a.shape[:2])

                sub_fb_d_1, sub_fb_d_2 = split_image_in_two(sub_fb_d)
                sub_fb_g_1, sub_fb_g_2 = split_image_in_two(sub_fb_g)
                sub_fb_a_1, sub_fb_a_2 = split_image_in_two(sub_fb_a)

                f_d_1, f_d_2 = split_image_in_two(f_d)
                f_g_1, f_g_2 = split_image_in_two(f_g)
                f_a_1, f_a_2 = split_image_in_two(f_a)

                cv.imwrite(str(self.out_images  / f'{directory.name}_0_subfb_d1.png'), sub_fb_d_1)
                cv.imwrite(str(self.out_images  / f'{directory.name}_1_subfb_d2.png'), sub_fb_d_2)
                cv.imwrite(str(self.out_images  / f'{directory.name}_2_subfb_g1.png'), sub_fb_g_1)
                cv.imwrite(str(self.out_images  / f'{directory.name}_3_subfb_g2.png'), sub_fb_g_2)
                cv.imwrite(str(self.out_images  / f'{directory.name}_4_subfb_a2.png'), sub_fb_a_2)

                cv.imwrite(str(self.out_images  / f'{directory.name}_0_fi_d1.png'), f_d_1)
                cv.imwrite(str(self.out_images /  f'{directory.name}_1_fi_d2.png'), f_d_2)
                cv.imwrite(str(self.out_images  / f'{directory.name}_2_fi_g1.png'), f_g_1)
                cv.imwrite(str(self.out_images /  f'{directory.name}_3_fi_g2.png'), f_g_2)
                cv.imwrite(str(self.out_images /  f'{directory.name}_4_fi_a2.png'), f_a_2)

                metadata = {'c_fi_size': sub_fb_d.shape[:2],
                            'o_fi_size': o_fi_size,
                            'c_subfb_size': sub_fb_d.shape[:2],
                            'o_fb_size': fb.shape[:2],
                            'downsampling': self.downsampling,
                            'fragment_size': fragment_size}

        logger.info('Image Matching Preprocessing: save metadata')
        save_preprocessing_meta(self.out_preprocessing, metadata)
        logger.info('Image Matching Preprocessing: End with success')








