#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Nghia Tran


"""
Detects Visualize output bounding boxes on images

Input: Image

Usage:
usage: visualize.py [-h] [--groundtruth GROUNDTRUTH] [--threshold THRESHOLD]
                    image_path outdir
"""

from __future__ import print_function

import random

from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
import scipy as scp
import argparse
import os
import sys
import logging
import matplotlib
import numpy as np

sys.path.insert(1, 'incl')

from utils import data_utils
from utils.annolist import AnnotationLib

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

parser = argparse.ArgumentParser(description='Create summsion for Kitti')
parser.add_argument('image_path', type=str, help='Path to test folder.')
parser.add_argument('outdirs', type=str, nargs = '*', help='Path to output txt files.')
parser.add_argument('--groundtruth', '-g', type=str, default=None, help='Path to groundtruth txt files.')
parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Confidence threshold.')

index = 0
car_pred = (0, 0, 255)
car_groundtruth = (255, 0, 0)


def create_proxy(color):
    color = np.array(color) / 255.0
    line = matplotlib.lines.Line2D([0,0], [10,10], color=color)
    return line

def read_rects(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    rects = []
    for line in lines:
        # 0    1 2 3  4  5  6  7 8 9 0 1 2 3 4  5
        # "Car 0 1 0 %f %f %f %f 0 0 0 0 0 0 0 %f"
        split = line.split(' ')

        # If not a car, don't care
        if split[0] != 'Car':
            continue

        x1 = float(split[4])
        y1 = float(split[5])
        x2 = float(split[6])
        y2 = float(split[7])
        rect = AnnotationLib.AnnoRect(x1, y1, x2, y2)
        rect.score = float(split[-1])
        rects.append(rect)
    return rects

def _draw_rect(draw, rect, color):
    left = rect.x1
    bottom = rect.y2
    right = rect.x2
    top = rect.y1
    rect_cords = ((left, top), (left, bottom),
                  (right, bottom), (right, top),
                  (left, top))
    draw.line(rect_cords, fill=color, width=3)

def main():
    global index
    args = parser.parse_args()
    image_path = args.image_path
    outdir = args.outdirs[0]
    files = sorted(os.listdir(outdir))

    colors=[]
    for i in range(len(args.outdirs)):
        colors.append((random.randrange(0,255), random.randrange(0,255), random.randrange(0,255)))

    if len(files) == 0:
        logging.error('No files found at %s' % (outdir))

    def get_data(direction=1):
        global index
        index = (index + direction) % len(files)
        filename, _ = os.path.splitext(files[index])
        image = scp.misc.imread(os.path.join(image_path, filename + '.png'))
        im = Image.fromarray(image.astype('uint8'))
        draw = ImageDraw.Draw(im)

        for i, outdir in enumerate(args.outdirs):
            rects = read_rects(os.path.join(outdir, filename + '.txt'))

            for rect in rects:
                if rect.score >= args.threshold:
                    _draw_rect(draw, rect, colors[i])

            if args.groundtruth is not None:
                rects = read_rects(os.path.join(args.groundtruth, filename + '.txt'))
                for rect in rects:
                    _draw_rect(draw, rect, car_groundtruth)

        return im

    def press(event):
        sys.stdout.flush()
        if event.key == 'x':
            im = get_data(1)
            ax.imshow(im)
            fig.canvas.draw()
        if event.key == 'z':
            im = get_data(-1)
            ax.imshow(im)
            fig.canvas.draw()

    fig, ax = plt.subplots()

    fig.canvas.mpl_connect('key_press_event', press)

    im = get_data()
    ax.imshow(im)
    ax.set_title('Press x to see next image, and z to see previous one.')


    proxies = [create_proxy(colors[i]) for i in range(len(args.outdirs))]
    ax.legend(proxies, args.outdirs, numpoints=1, markerscale=2)
    plt.show()

if __name__ == '__main__':
    main()
