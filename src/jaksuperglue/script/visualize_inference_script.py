import typer
import json
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import ConnectionPatch

app = typer.Typer()


def load_control_points(control_points_path: str):
    with open(str(control_points_path), 'r') as f:
        control_points = json.load(f)

    return control_points


@app.command()
def visualize_inference(control_points_path: str,
                        images_path: str):
    control_points = load_control_points(control_points_path)

    images_path = Path(images_path)
    f_d = cv.imread(str(images_path / 'd.tiff'))
    f_g = cv.imread(str(images_path / 'g.tiff'))
    f_a = cv.imread(str(images_path / 'a.tiff'))
    fb = cv.imread(str(images_path / f'{images_path.name}.png'), 0)

    kpts_fi_d = np.array(control_points["d.tiff"])
    kpts_fi_g = np.array(control_points["g.tiff"])
    kpts_fi_a = np.array(control_points["a.tiff"])

    kpts_fb_d = np.array(control_points["fakebubble_d"])
    kpts_fb_g = np.array(control_points["fakebubble_g"])
    kpts_fb_a = np.array(control_points["fakebubble_a"])

    conf_d = np.array(control_points["confidence_d"])
    conf_g = np.array(control_points["confidence_g"])
    conf_a = np.array(control_points["confidence_a"])

    kpts_fb_list = [kpts_fb_d, kpts_fb_g, kpts_fb_a]
    kpts_fi_list = [kpts_fi_d, kpts_fi_g, kpts_fi_a]
    conf_list = [conf_d, conf_g, conf_a]

    fisheye_list = [f_d, f_g, f_a]

    for idx in range(3):
        color = cm.jet(conf_list[idx])
        dpi = 100
        size = 6
        pad = .5
        n = 2
        figsize = (size * n, size * 3 / 4) if size is not None else None
        fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
        ax[0].imshow(fisheye_list[idx][:, :, ::-1])
        ax[0].get_yaxis().set_ticks([])
        ax[0].get_xaxis().set_ticks([])
        for spine in ax[0].spines.values():  # remove frame
            spine.set_visible(False)

        ax[0].scatter(kpts_fi_list[idx][:, 0], kpts_fi_list[idx][:, 1], c=color, s=2)


        ax[1].imshow(fb, cmap='gray')
        ax[1].get_yaxis().set_ticks([])
        ax[1].get_xaxis().set_ticks([])
        for spine in ax[1].spines.values():  # remove frame
            spine.set_visible(False)

        ax[1].scatter(kpts_fb_list[idx][:, 0], kpts_fb_list[idx][:, 1], c=color, s=2)

        for i in range(kpts_fi_list[idx].shape[0]):
            xyA = tuple(kpts_fi_list[idx][i])
            xyB = tuple(kpts_fb_list[idx][i])
            con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                                  axesA=ax[0], axesB=ax[1], color=color[i])
            fig.add_artist(con)

        plt.tight_layout(pad=pad)
        plt.show()

if __name__ == "__main__":
    app()
