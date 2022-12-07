from loguru import logger
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import datetime
import matplotlib.cm as cm
from jaksuperglue.models.utils import (make_matching_plot,
                                       read_image)
from jaksuperglue.dataset.base_dataset import JakOnlyImageDataset, JakOnlyImageCollator
from jaksuperglue.models.matching import Matching
from jaksuperglue.utils.evaluation_utils import get_score, type2int


class EvaluateEngine:
    def __init__(self,
                 eval_path: str,
                 model_config: dict,
                 dataset_config: dict,
                 batch_size: int = 1,
                 nb_workers: int = 2,
                 device: str = 'cuda'):

        self.device = device
        self.dataset_config = dataset_config
        self.model = Matching(model_config).eval().to(device)
        self.eval_set = JakOnlyImageDataset(eval_path, mode='eval')
        self.eval_loader = DataLoader(self.eval_set, batch_size=batch_size, num_workers=nb_workers,
                                      collate_fn=JakOnlyImageCollator(device), shuffle=True)

    def evaluate(self):
        logger.info("Starting evaluation")
        self.model.eval()
        thresh_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = []

        with torch.no_grad():
            for sample_id, x in enumerate(tqdm(self.eval_loader)):
                im_type = x['type']
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

                mean_list = []
                std_list = []
                kpts_list = []

                for count, thresh in enumerate(thresh_list):
                    mean, std, nb_kpts = get_score(mkpts0,
                                                   mkpts1,
                                                   mconf,
                                                   im_type,
                                                   self.dataset_config,
                                                   thresh,
                                                   x['image1'].cpu().numpy().squeeze())

                    result.append({
                        'sample_id': sample_id,
                        'im_type': im_type,
                        'threshold': thresh,
                        'nb_kpts': nb_kpts,
                        'precision': mean,
                    })

        df = pd.DataFrame(result)
        df.to_pickle(f"../../output/results_magicleappretrained_{str(datetime.date.today()).replace('-', '')}.pickle")

        return df
