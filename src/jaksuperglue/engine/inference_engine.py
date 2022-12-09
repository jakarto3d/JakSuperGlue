from torch.utils.data import DataLoader
from jaksuperglue.dataset.base_dataset import JakInferDataset, JakInferCollator
from jaksuperglue.models.matching import Matching
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

from jaksuperglue.utils.inference_utils import write_keypoints


class InferenceEngine:
    def __init__(self,
                 working_dir: str,
                 output_path: str,
                 dataset_config: dict,
                 batch_size: int = 1,
                 nb_workers: int = 2,
                 max_keypoints: int = 1024,
                 match_threshold: int = 0.5,
                 device: str = 'cuda'):

        self.device = device
        self.output_path = output_path
        self.dataset_config = dataset_config

        model_config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': match_threshold,
            }
        }
        self.model = Matching(model_config).eval().to(device)
        self.infer_set = JakInferDataset(working_dir, mode='eval')
        self.infer_loader = DataLoader(self.infer_set, batch_size=batch_size, num_workers=nb_workers,
                                      collate_fn=JakInferCollator(device), shuffle=True)

    def infer(self):
        logger.info("Image Matching Inference: Start")
        self.model.eval()

        with torch.no_grad():
            for sample_id, x in enumerate(tqdm(self.infer_set)):
                im_type = x['type']
                sphere_name = x['sphere_name']

                # Perform the matching.
                pred = self.model(x)
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']

                # Keep the matching keypoints.
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]

                write_keypoints(mkpts0, mkpts1, mconf, im_type, sphere_name, self.output_path)

        logger.info("Image Matching Inference: End with success")
                # # Visualize the matches.
                # color = cm.jet(mconf)
                # text = [
                #     'SuperGlue',
                #     'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                #     'Matches: {}'.format(len(mkpts0)),
                # ]
                #
                #
                # make_matching_plot(
                #     x['image0'].cpu().numpy().squeeze()*255, x['image1'].cpu().numpy().squeeze()*255, kpts0, kpts1, mkpts0, mkpts1, color,
                #     text)
