from pathlib import Path
from typing import Union, Tuple
import cv2 as cv
from jaksuperglue.utils.dataset_utils import load_sphere, load_fisheye, get_sub_fb_d, get_sub_fb_g, get_sub_fb_a, \
    get_fakebubble_fragment, split_image_in_two


class PreProcessingEngine:
    def __init__(self,
                 working_dir: Union[Path, str],
                 downsampling: int = 2,
                 nb_fragment: Tuple[int, int] = (8, 16)):
        self.working_dir = Path(working_dir)
        self.temp_directory = Path(self.working_dir.parents[0] / 'temp_superglue')
        self.downsampling = downsampling
        self.nb_fragment = nb_fragment

    def process(self):
        for directory in self.working_dir.iterdir():
            if directory.is_dir():
                fb_path = str(directory / Path(str(directory.name) + '.png'))
                f_d_path = str(directory / 'd.tiff')
                f_g_path = str(directory / 'g.tiff')
                f_a_path = str(directory / 'a.tiff')
                fb = load_sphere(fb_path, self.downsampling)

                f_d = load_fisheye(f_d_path, resize=fb.shape[:2])
                f_g = load_fisheye(f_g_path, resize=fb.shape[:2])
                f_a = load_fisheye(f_a_path, resize=fb.shape[:2])

                fakebubble_fragments, _ = get_fakebubble_fragment(fb, self.nb_fragment[0], self.nb_fragment[1])

                sub_fb_d = get_sub_fb_d(fakebubble_fragments, self.downsampling)
                sub_fb_g = get_sub_fb_g(fakebubble_fragments, self.downsampling)
                sub_fb_a = get_sub_fb_a(fakebubble_fragments, self.downsampling)

                sub_fb_d_1, sub_fb_d_2 = split_image_in_two(sub_fb_d)
                sub_fb_g_1, sub_fb_g_2 = split_image_in_two(sub_fb_g)
                sub_fb_a_1, sub_fb_a_2 = split_image_in_two(sub_fb_a)

                f_d_1, f_d_2 = split_image_in_two(f_d)
                f_g_1, f_g_2 = split_image_in_two(f_g)
                f_a_1, f_a_2 = split_image_in_two(f_a)

                cv.imwrite(str(self.temp_directory  / f'{directory.name}_0_subfb_d1.png'), sub_fb_d_1)
                cv.imwrite(str(self.temp_directory  / f'{directory.name}_1_subfb_d2.png'), sub_fb_d_2)
                cv.imwrite(str(self.temp_directory  / f'{directory.name}_2_subfb_g1.png'), sub_fb_g_1)
                cv.imwrite(str(self.temp_directory  / f'{directory.name}_3_subfb_g2.png'), sub_fb_g_2)
                cv.imwrite(str(self.temp_directory  / f'{directory.name}_4_subfb_a2.png'), sub_fb_a_2)

                cv.imwrite(str(self.temp_directory  / f'{directory.name}_0_fi_d1.png'), f_d_1)
                cv.imwrite(str(self.temp_directory /  f'{directory.name}_1_fi_d2.png'), f_d_2)
                cv.imwrite(str(self.temp_directory  / f'{directory.name}_2_fi_g1.png'), f_g_1)
                cv.imwrite(str(self.temp_directory /  f'{directory.name}_3_fi_g2.png'), f_g_2)
                cv.imwrite(str(self.temp_directory /  f'{directory.name}_4_fi_a2.png'), f_a_2)







