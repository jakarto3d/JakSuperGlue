import typer
from scipy.spatial.transform import Rotation as R
import torch
from jaksuperglue.engine.evaluate_engine import EvaluateEngine
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
app = typer.Typer()

def select_device(gpu: bool = True):
    if gpu:
        return "cuda"
    return "cpu"

@app.command()
def evaluate(eval_path: str,
             gpu: bool = True,
             batch_size: int = 1,
             nb_workers: int = 2,
             ):

    device = select_device(gpu=gpu)

    model_config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.3,
        }
    }

    dataset_config = {
        'downsampling': 2,
        'patch_size': (256, 256),
        'calib_d_path': '/home/hugo/Jakarto/Jaksphere/data/Calib/calibration_fisheye_right_17052022_094555.json',
        'calib_g_path': '/home/hugo/Jakarto/Jaksphere/data/Calib/calibration_fisheye_left_16052022_175041.json',
        'calib_a_path': '/home/hugo/Jakarto/Jaksphere/data/Calib/calibration_fisheye_back_17052022_095123.json',
        'rotation_d': R.from_euler('yzx', [-88.5276023, -3.6436489, -20.81274975], degrees=True).as_matrix(),
        'rotation_g': R.from_euler('yzx', [89.90079489, 1.28823787, -20.58384564], degrees=True).as_matrix(),
        'rotation_a': R.from_euler('yzx', [-93.94631351, -47.18336904, -88.81668717], degrees=True).as_matrix()
    }

    my_training_engine = EvaluateEngine(eval_path=eval_path,
                                        model_config=model_config,
                                        dataset_config=dataset_config,
                                        batch_size=batch_size,
                                        nb_workers=nb_workers,
                                        device=device)

    matrix_score_mean, matrix_score_mean_nan, matrix_score_std, matrix_score_std_nan, matrix_nb_kpts = my_training_engine.evaluate()



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    app()