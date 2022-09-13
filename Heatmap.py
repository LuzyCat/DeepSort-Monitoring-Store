import os
import sys
import numpy as np
import cv2
import math
import argparse
from scipy import spatial
from scipy.ndimage.filters import gaussian_filter 
from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt
import pathlib
from sklearn.preprocessing import MinMaxScaler

matplotlib.use('TKAgg')

from iutils import xywh_to_foot

def grid_density_kdtree(xl, yl, xi, yi, dfactor):
    zz = np.empty([len(xi),len(yi)], dtype=np.uint8)
    zipped = zip(xl, yl)
    kdtree = spatial.KDTree(zipped)
    for xci in range(0, len(xi)):
        xc = xi[xci]
        for yci in range(0, len(yi)):
            yc = yi[yci]
            density = 0.
            retvalset = kdtree.query((xc,yc), k=5)
            for dist in retvalset[0]:
                density = density + math.exp(-dfactor * pow(dist, 2)) / 5
            zz[yci][xci] = min(density, 1.0) * 255
    return zz

def makeHeatmap(imgpth, txtpth, hist=7):
    path = pathlib.Path(imgpth)
    img = cv2.imread(imgpth)
    width, height = img.shape[1], img.shape[0]
    
    file_annot = open(txtpth, 'r')
    lines = file_annot.readlines()
    file_annot.close()
    boxes = []
    coords = []
    for line in lines:
        res = line.split(' ')
        box = [int(x) for x in res[3:7]] # left, top, w, h
        ct = xywh_to_foot(box, width, height)
        if ct[1] < height -1:
            boxes.append(box) 
            coords.append(ct)
    boxes = np.asarray(boxes)
    coords = np.asarray(coords)
    
    density = np.zeros(img.shape, dtype=np.float32)
    counts = coords.shape[0]
    leafsize = 2048
    tree = spatial.KDTree(coords, leafsize)
    distances, locations = tree.query(coords, k=4)
    for i, pt in enumerate(coords):
        pt2d = np.zeros(img.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if counts > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(img.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    # plt.imshow(density, cmap=CM.jet)
    # max = distances.max() * 3 * 0.1
    K = gaussian_filter(density, hist)
    norm_image = cv2.normalize(K, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)

    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    # K = plt.imshow(K, cmap=plt.cm.get_cmap("jet"))
    # K8 = ((K/K.max()) * 255).astype(np.uint8)
    heatmap_img = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 1.0, heatmap_img, 0.7, 0)
    newname = 'Heatmap_' + path.stem
    cv2.imwrite(str(path.parent / newname) + '.png', img)
    # cv2.imshow(newname, img)
    print('Heatmap Done.')
    # if cv2.waitKey(0) == ord('q'):
    #     return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='./runs/track/crowdhuman_yolov5m_osnet_x1_0_market150120/0.png')
    parser.add_argument('--txt', type=str, default='./runs/track/crowdhuman_yolov5m_osnet_x1_0_market150120/tracks/0.txt')
    args = parser.parse_args()
    makeHeatmap(args.img, args.txt)