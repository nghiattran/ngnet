from __future__ import print_function

from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
import scipy as scp
import argparse
import os
import sys
import logging
import numpy as np

sys.path.insert(1, 'incl')

from utils import data_utils
from utils.annolist import AnnotationLib

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

parser = argparse.ArgumentParser(description='Create summsion for Kitti')
parser.add_argument('image_path', type=str, help='Path to test folder.')
parser.add_argument('outdir', type=str, help='Path to output txt files.')
parser.add_argument('--groundtruth', '-g', type=str, default=None, help='Path to groundtruth txt files.')
parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Confidence threshold.')

index = 0
outdir = ''
files = []
car_pred = (0, 0, 255)
car_groundtruth = (255, 0, 0)

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
    draw.line(rect_cords, fill=color, width=2)

def main():
    global index, files, outdir
    args = parser.parse_args()
    image_path = args.image_path
    outdir = args.outdir
    files = sorted(os.listdir(outdir))

    if len(files) == 0:
        logging.error('No files found at %s' % (outdir))

    def get_data(direction=1):
        global index, files, outdir
        index = (index + direction) % len(files)
        filename, _ = os.path.splitext(files[index])
        image = scp.misc.imread(os.path.join(image_path, filename + '.png'))

        rects = read_rects(os.path.join(outdir, filename + '.txt'))
        im = Image.fromarray(image.astype('uint8'))
        draw = ImageDraw.Draw(im)

        for rect in rects:
            if rect.score >= args.threshold:
                _draw_rect(draw, rect, car_pred)

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
    plt.show()

if __name__ == '__main__':
    main()
