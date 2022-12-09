import json
from os import mkdir

import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from jaksuperglue.utils.inference_utils import back2original_fi_1, back2original_fi_2, back2original_subfb_d1, \
    back2original_subfb_d2, back2original_subfb_g1, back2original_subfb_g2, back2original_subfb_a2, load_keypoints, \
    load_preprocessing_meta


class PostProcessingEngine:
    def __init__(self,
                 predictions_path: str,
                 preprocessing_meta_path: str,
                 output_path: str):
        self.prediction_path = Path(predictions_path)
        self.preprocessing_meta_path = preprocessing_meta_path
        self.prediction_files = list(self.prediction_path.glob('*.json'))
        self.sphere_name_list = list(set([file.name.split('_')[0] for file in self.prediction_files]))
        self.output_path = Path(output_path)
        self.im_type = ['d1', 'd2', 'g1', 'g2', 'a2']

    def process(self):
        logger.info('Image Matching Postprocessing: start')
        mkdir(str(self.output_path / 'control_points'))

        meta = load_preprocessing_meta(self.preprocessing_meta_path)
        for sphere_name in tqdm(self.sphere_name_list):
            d1_path = str(self.prediction_path / f'{sphere_name}_d1.json')
            d2_path = str(self.prediction_path / f'{sphere_name}_d2.json')
            g1_path = str(self.prediction_path / f'{sphere_name}_g1.json')
            g2_path = str(self.prediction_path / f'{sphere_name}_g2.json')
            a2_path = str(self.prediction_path / f'{sphere_name}_a2.json')

            kpts_fi_d1, kpts_subfb_d1, conf_d1 = load_keypoints(d1_path)
            kpts_fi_d2, kpts_subfb_d2, conf_d2 = load_keypoints(d2_path)
            kpts_fi_g1, kpts_subfb_g1, conf_g1 = load_keypoints(g1_path)
            kpts_fi_g2, kpts_subfb_g2, conf_g2 = load_keypoints(g2_path)
            kpts_fi_a2, kpts_subfb_a2, conf_a2 = load_keypoints(a2_path)

            kpts_fi_d1 = back2original_fi_1(kpts_fi_d1, meta)
            kpts_fi_d2 = back2original_fi_2(kpts_fi_d2, meta)
            kpts_fi_g1 = back2original_fi_1(kpts_fi_g1, meta)
            kpts_fi_g2 = back2original_fi_2(kpts_fi_g2, meta)
            kpts_fi_a2 = back2original_fi_2(kpts_fi_a2, meta)
            kpts_subfb_d1 = back2original_subfb_d1(kpts_subfb_d1, meta)
            kpts_subfb_d2 = back2original_subfb_d2(kpts_subfb_d2, meta)
            kpts_subfb_g1 = back2original_subfb_g1(kpts_subfb_g1, meta)
            kpts_subfb_g2 = back2original_subfb_g2(kpts_subfb_g2, meta)
            kpts_subfb_a2 = back2original_subfb_a2(kpts_subfb_a2, meta)

            kpts_fi_d = np.vtack((kpts_fi_d1, kpts_fi_d2))
            kpts_fi_g = np.vtack((kpts_fi_g1, kpts_fi_g2))
            kpts_fb_d = np.vstack((kpts_subfb_d1, kpts_subfb_d2))
            kpts_fb_g = np.vstack((kpts_subfb_g1, kpts_subfb_g2))

            confidence_d = np.hstack((conf_d1, conf_d2))
            confidence_g = np.hstack((conf_g1, conf_g2))

            with open(str(self.output_path / 'control_points' / f'{sphere_name}.json')) as f:
                control_points = {"d.tiff": kpts_fi_d.tolist(),
                                  "g.tiff": kpts_fi_g.tolist(),
                                  "a.tiff": kpts_fi_a2.tolist(),
                                  "fakebubble_d": kpts_fb_d.tolist(),
                                  "fakebubble_g": kpts_fb_g.tolist(),
                                  "fakebubble_a": kpts_subfb_a2.tolist(),
                                  "confidence_d": confidence_d.tolist(),
                                  "confidence_g": confidence_g.tolist(),
                                  "confidence_a": conf_a2.tolist()}

                json.dump(control_points, f)

        logger.info('Image Matching Postprocessing: end with success')
