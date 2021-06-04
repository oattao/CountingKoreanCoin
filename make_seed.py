import os
import glob
import shutil
import cv2 as cv
import numpy as np
import argparse

# parse input parameter
parser = argparse.ArgumentParser()
parser.add_argument('--input_path')
parser.add_argument('--output_path')
parser.add_argument('--thresh', type=int, default=120)
parser.add_argument('--morph_iteration', type=int, default=3)
args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path
thresh = args.thresh
morph_iteration = args.morph_iteration

# prepare output foloder
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.mkdir(output_path)

file_list = glob.glob(f'{input_path}/*.jpg')
for fname in file_list:
    print('Processing file: ', fname)
    basename = os.path.basename(fname)
    img = cv.imread(fname)
    gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, timg = cv.threshold(gimg, thresh=thresh, maxval=255, type=cv.THRESH_BINARY_INV)
    kernel = np.ones((3, 3))
    oimg = cv.morphologyEx(timg, cv.MORPH_OPEN, kernel, iterations=morph_iteration)
    contours, _ = cv.findContours(oimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue
    bcnt = max(contours, key=cv.contourArea)
    (x, y), r = cv.minEnclosingCircle(bcnt)
    x, y, r = map(int, [x, y, r])
    mask = np.zeros_like(gimg)
    cv.circle(mask, (x, y), r, (255), -1)
    nimg = cv.bitwise_and(img, img, mask=mask)
    xmin, xmax = x - r, x + r
    ymin, ymax = y - r, y + r
    coin = nimg[ymin: ymax, xmin: xmax, :]
    cv.imwrite(os.path.join(output_path, basename), coin)
