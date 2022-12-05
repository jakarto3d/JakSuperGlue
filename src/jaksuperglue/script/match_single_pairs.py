import matplotlib.cm as cm
import torch


from src.jaksuperglue.models.matching import Matching
from src.jaksuperglue.models.utils import (make_matching_plot,
                                           read_image)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    # im1_path = '/home/hugo/TEMPORAIRE/temp/fakebubbles/jak2_20220615_78824s446ms/jak2_20220615_78824s446ms.png'
    # im2_path = '/home/hugo/TEMPORAIRE/temp/fakebubbles/jak2_20220615_78824s446ms/g.tiff'
    # im1_path = '/home/hugo/Jakarto/Jakarto3D_KeypointsPointsMatching/template_41/jak2_20210114_55101s606ms/jak2_20210114_55101s606ms.png'
    # im2_path = '/home/hugo/Jakarto/Jakarto3D_KeypointsPointsMatching/template_41/jak2_20210114_55101s606ms/d.tiff'
    im1_path = '/home/hugo/TEMPORAIRE/image_matching/sample_1/img_0.jpg'
    im2_path = '/home/hugo/TEMPORAIRE/image_matching/sample_1/img_2.jpg'
    device = 'cuda'
    ds = 4
    # resize = (8192 // ds, 4096 // ds)
    resize = (1024, 1024)
    resize_float = True

    # Load the image pair.
    image0, inp0, scales0 = read_image(
        im1_path, device, resize, 0, resize_float)
    image1, inp1, scales1 = read_image(
        im2_path, device, resize, 0, resize_float)

    weights = ['indoor', 'outdoor']

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': weights[1],
            'sinkhorn_iterations': 20,
            'match_threshold': 0.4,
        }
    }

    matching = Matching(config).eval().to(device)

    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    # Visualize the matches.
    color = cm.jet(mconf)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]

    # Display extra parameter info.
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
    ]

    make_matching_plot(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
        text, small_text=small_text)