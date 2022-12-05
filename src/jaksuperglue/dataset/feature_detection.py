import cv2 as cv
import numpy as np

def get_edge_sobel(src: np.array):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    src = cv.GaussianBlur(src, (3, 3), 0)

    grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    return cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def get_sobel_keypoints(img_rgb: np.array,
                        nb_kpts_patch: int):
    im_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    gradient = get_edge_sobel(im_gray)

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    kp_im = fast.detect(gradient, None)

    u = [kp.pt[1] for kp in kp_im]
    v = [kp.pt[0] for kp in kp_im]
    score = [kp.response for kp in kp_im]
    uv_point = np.vstack((u, v)).T
    index_best = np.argsort(np.array(score))[::-1][:nb_kpts_patch]
    best_uv_point = np.round(uv_point[index_best])

    return best_uv_point


def get_sift_keypoints(img_rgb: np.array,
                       nb_kpts_patch: int):
    sift = cv.SIFT_create()

    # Compute sift points
    im_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    kp_im, des = sift.detectAndCompute(im_gray, None)

    # Extract best points in im_1
    u = [kp.pt[1] for kp in kp_im]
    v = [kp.pt[0] for kp in kp_im]
    uv_point = np.vstack((u, v)).T
    score = [kp.response for kp in kp_im]
    size = np.array([kp.size for kp in kp_im])
    index_size = np.where(size > 4.5)[0]
    index_best = np.argsort(np.array(score)[index_size])[::-1][:nb_kpts_patch]
    best_uv_point = np.round(uv_point[index_size][index_best])

    return best_uv_point