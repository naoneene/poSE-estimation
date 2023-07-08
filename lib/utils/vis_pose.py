# -*-coding:utf-8-*-

import cv2
import numpy as np
import matplotlib.pyplot as plt


def vis_keypoints_2D_crop(coord_2d, img):

    img_base = np.zeros(img.shape, np.uint8)

    connections = [[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14],
                   [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]

    coord_2d = coord_2d * 256

    for ind, (i, j) in enumerate(connections):
        x = (coord_2d[i, 0], coord_2d[i, 1])
        y = (coord_2d[j, 0], coord_2d[j, 1])
        cv2.line(img_base, x, y, (255, 250, 0), 2, cv2.LINE_AA)

    for i in range(17):
        if (i is 0) or (i is 7) or (i is 8) or (i is 9) or (i is 10):
            cv2.circle(img_base, (coord_2d[i, 0], coord_2d[i, 1]), 5, (0, 255, 0), -1, cv2.LINE_AA)
        elif (1 <= i <= 3) or (14 <= i <= 16):
            cv2.circle(img_base, (coord_2d[i, 0], coord_2d[i, 1]), 5, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(img_base, (coord_2d[i, 0], coord_2d[i, 1]), 5, (255, 0, 0), -1, cv2.LINE_AA)

    img = cv2.addWeighted(img, 1.0, img_base, 0.8, 0)

    plt.imshow(img)

    return 0


def vis_keypoints_2D(ret, coord_2d, resize_info, img, lock_num):

    img_base = np.zeros(img.shape, np.uint8)

    connections = [[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14],
                   [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]

    coord_2d = coord_2d * 256

    shift = 20
    for i in range(lock_num):
        for j in range(17):
            coord_2d[i, j, 0] = int(coord_2d[i, j, 0] * resize_info[i, 3] + ret[i, 0] - shift - resize_info[i, 1])
            coord_2d[i, j, 1] = int(coord_2d[i, j, 1] * resize_info[i, 2] + ret[i, 1] - shift - resize_info[i, 0])

    for i in range(lock_num):
        for ind, (j, k) in enumerate(connections):
            x = (coord_2d[i, j, 0], coord_2d[i, j, 1])
            y = (coord_2d[i, k, 0], coord_2d[i, k, 1])
            cv2.line(img_base, x, y, (0, 250, 250), 2, cv2.LINE_AA)

    for i in range(lock_num):
        if img.shape[0] is 256:
            plot_size = 5
        else:
            if (ret[i, 3] - ret[i, 1]) > 250:
                plot_size = 6
            else:
                plot_size = 4

        for j in range(17):
            if (j is 0) or (j is 7) or (j is 8) or (j is 9) or (j is 10):
                cv2.circle(img_base, (coord_2d[i, j, 0], coord_2d[i, j, 1]), plot_size, (0, 255, 0), -1, cv2.LINE_AA)
            elif (1 <= j <= 3) or (14 <= j <= 16):
                cv2.circle(img_base, (coord_2d[i, j, 0], coord_2d[i, j, 1]), plot_size, (255, 0, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(img_base, (coord_2d[i, j, 0], coord_2d[i, j, 1]), plot_size, (0, 0, 255), -1, cv2.LINE_AA)

    # cv2.imshow('2D skeleton', img_base)

    img = cv2.addWeighted(img, 1.0, img_base, 1.0, 0)

    return img


def vis_keypoints_2D_h36m(coord_2d, img):

    img_base = np.zeros(img.shape, np.uint8)

    connections = [[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14],
                   [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]

    coord_2d = coord_2d * 256

    for ind, (i, j) in enumerate(connections):
        x = (coord_2d[i, 0], coord_2d[i, 1])
        y = (coord_2d[j, 0], coord_2d[j, 1])
        cv2.line(img_base, x, y, (250, 250, 0), 2, cv2.LINE_AA)

    plot_size = 4
    for i in range(17):
        if (i is 0) or (i is 7) or (i is 8) or (i is 9) or (i is 10):
            cv2.circle(img_base, (coord_2d[i, 0], coord_2d[i, 1]), plot_size, (0, 255, 0), -1, cv2.LINE_AA)
        elif (1 <= i <= 3) or (14 <= i <= 16):
            cv2.circle(img_base, (coord_2d[i, 0], coord_2d[i, 1]), plot_size, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(img_base, (coord_2d[i, 0], coord_2d[i, 1]), plot_size, (255, 0, 0), -1, cv2.LINE_AA)

    # cv2.imshow('2D skeleton', img_base)

    img = cv2.addWeighted(img, 1.0, img_base, 0.7, 0)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)

    return img


def vis_keypoints_2D_projection(coord_2d, img):

    img_base = np.zeros(img.shape, np.uint8)
    img_base.fill(255)

    connections = [[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14],
                   [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]

    for ind, (i, j) in enumerate(connections):
        x = (coord_2d[i, 0], coord_2d[i, 1])
        y = (coord_2d[j, 0], coord_2d[j, 1])
        cv2.line(img_base, x, y, (250, 250, 0), 2, cv2.LINE_AA)

    plot_size = 4
    for i in range(17):
        if (i is 0) or (i is 7) or (i is 8) or (i is 9) or (i is 10):
            cv2.circle(img_base, (coord_2d[i, 0], coord_2d[i, 1]), plot_size, (0, 255, 0), -1, cv2.LINE_AA)
        elif (1 <= i <= 3) or (14 <= i <= 16):
            cv2.circle(img_base, (coord_2d[i, 0], coord_2d[i, 1]), plot_size, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(img_base, (coord_2d[i, 0], coord_2d[i, 1]), plot_size, (255, 0, 0), -1, cv2.LINE_AA)

    # cv2.imshow('2D skeleton', img_base)

    return img_base


def vis_keypoints_3D(vals, front, side):

    connections = [[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14],
                   [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]

    LR = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    lcolor = '#AA0000'
    rcolor = '#00008B'

    for ind, (i, j) in enumerate(connections):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        if (ind is 0) or (ind is 1) or (ind is 2) or (ind is 3):
            front.plot(x, -y, z, alpha=0.8, lw=3, c='#F37620', marker='o', markersize=5, markerfacecolor='#F37620')
            side.plot(x, z, -y, alpha=0.8, lw=3, c='#F37620', marker='o', markersize=5, markerfacecolor='#F37620')
        else:
            front.plot(x, -y, z, alpha=0.6, lw=3, c=lcolor if LR[ind] else rcolor, marker='o', markersize=5,
                    markerfacecolor=lcolor if LR[ind] else rcolor)
            side.plot(x, z, -y, alpha=0.6, lw=3, c=lcolor if LR[ind] else rcolor, marker='o', markersize=5,
                     markerfacecolor=lcolor if LR[ind] else rcolor)

    # ---------------------------------------------------------------------

    front.set_title('Front View', fontsize=12)

    front.patch.set_facecolor('#FFFFFF')
    front.patch.set_alpha(0.1)

    plt.rcParams["axes.edgecolor"] = "#000000"
    plt.rc('grid', alpha=0.2, lw=1, linestyle="-", c='#AFAFAF')
    front.grid(True)

    front.set_xlim3d([-1000, 1000])
    front.set_ylim3d([-1000, 1000])
    front.set_zlim3d([-1200, 1200])

    front.set_xticks(range(-1000, 1000, 250))
    front.set_yticks(range(-1000, 1000, 250))
    front.set_zticks(range(-1200, 1200, 300))

    plt.setp(front.get_xticklabels(), visible=False)
    plt.setp(front.get_yticklabels(), visible=False)
    plt.setp(front.get_zticklabels(), visible=False)
    front.tick_params(axis='both', which='major', length=0)

    front.set_xlabel('X', fontsize=10)
    front.set_ylabel('Y', fontsize=10)
    front.set_zlabel('Z', fontsize=10)

    front.xaxis._axinfo['juggled'] = (2, 0, 1)
    front.yaxis._axinfo['juggled'] = (1, 1, 1)
    front.zaxis._axinfo['juggled'] = (2, 2, 2)

    front.view_init(elev=115, azim=-90)

    # ---------------------------------------------------------------------

    side.set_title('Side View', fontsize=12)

    side.patch.set_facecolor('#FFFFFF')
    side.patch.set_alpha(0.1)

    plt.rcParams["axes.edgecolor"] = "#000000"
    plt.rc('grid', alpha=0.2, lw=1, linestyle="-", c='#AFAFAF')
    side.grid(True)

    side.set_xlim3d([-1000, 1000])
    side.set_ylim3d([-1200, 1200])
    side.set_zlim3d([-1000, 1000])

    side.set_xticks(range(-1000, 1000, 250))
    side.set_yticks(range(-1200, 1200, 300))
    side.set_zticks(range(-1000, 1000, 250))

    plt.setp(side.get_xticklabels(), visible=False)
    plt.setp(side.get_yticklabels(), visible=False)
    plt.setp(side.get_zticklabels(), visible=False)
    side.tick_params(axis='both', which='major', length=0)

    side.set_xlabel('X', fontsize=10)
    side.set_ylabel('Z', fontsize=10)
    side.set_zlabel('Y', fontsize=10)

    side.xaxis._axinfo['juggled'] = (0, 0, 0)
    side.yaxis._axinfo['juggled'] = (0, 1, 2)
    side.zaxis._axinfo['juggled'] = (2, 2, 2)

    side.view_init(elev=25, azim=0)

    return 0

