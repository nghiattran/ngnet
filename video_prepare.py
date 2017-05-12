import os

import cv2
import imageio
import argparse
from PIL import Image, ImageDraw
from matplotlib import pyplot
import scipy.misc


parser = argparse.ArgumentParser(description='Detect cars in video.')
parser.add_argument('video', type=str, help='Path to video file.')
# parser.add_argument('logdir', type=str, help='Path to logdir.')
if __name__ == '__main__':
    args = parser.parse_args()

    filename = os.path.basename(args.video)
    save_file = 'new_' + filename

    vidcap = cv2.VideoCapture(args.video)
    cnt = 0
    with imageio.get_writer(save_file, mode='I', fps=20) as writer:
        while True:
            success, image = vidcap.read()
            if not success:
                break

            shape = image.shape
            oy = 500
            ox = (shape[1] - 1248) / 2
            print(oy,oy + 384, ox, ox+1248)
            image = image[oy:oy + 384, ox:ox+1248, :]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(image.shape)
            writer.append_data(image)
            cnt += 1

            if cnt == 20:
                break

# def draw(gh, gw, img):
#     im = Image.fromarray(img)
#     dr = ImageDraw.Draw(im)
#     H, W, _ = img.shape
#     w = W / gw
#     h = H / gh
#     for x in range(gw):
#         for y in range(gh):
#             for i in range(3):
#                 dr.rectangle(((x * w - i, y * h - i), ((x + 1) * w - i, (y + 1) * h - i)), outline="blue")
#     return im, dr
#
# img = scipy.misc.imread('/home/nghia/workplace/projects/machine_learning/KittiBox/DATA/Udacity/training/image_2/1478900143012608689.jpg')
# H, W, _ = img.shape
#
# im = Image.fromarray(img)
# dr = ImageDraw.Draw(im)
# with open('/home/nghia/workplace/projects/machine_learning/KittiBox/DATA/Udacity/training/label_2/1478900143012608689.txt', 'r') as f:
#     objs = f.readlines()
#     for obj in objs:
#         # Car 0.0 0 -1 1100.0 579.0 1614.0 775.0 -1 -1 -1 -1 -1 -1 -1
#         data = obj.split(' ')
#         if data[0] != 'Car':
#             continue
#
#         x1 = float(data[4])
#         y1 = float(data[5])
#         x2 = float(data[6])
#         y2 = float(data[7])
#         for i in range(3):
#             dr.rectangle(((x1 - i, y1 - i), (x2 - i, y2 - i)), outline="red")
# im.save('didi-gt.jpg')
#
# h, w = 600, 960
# img = scipy.misc.imresize(img, (h, w, 3))
# gh = 15
# gw = 24
# im, dr = draw(gh, gw, img)
# im.save('didi.jpg')
#
# img = scipy.misc.imread('/home/nghia/workplace/projects/machine_learning/KittiBox/DATA/KittiBox/training/image_2/000000.png')
# img = scipy.misc.imresize(img, (384, 1248, 3))
# gh = 12
# gw = 39
# im, _ = draw(gh, gw, img)
# im.save('kitti.jpg')