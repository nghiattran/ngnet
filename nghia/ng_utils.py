import matplotlib
import scipy.misc
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.util


path = '/home/nghia/workplace/data/new_kitti_crowdai/training/image_2/'
images = sorted(os.listdir(path))
index = -1

def next_image():
    global images, index
    index = (index + 1) % len(images)
    if images[index].endswith('.txt'):
        return next_image()

    img = scipy.misc.imread(os.path.join(path, images[index]))
    noisy = skimage.util.random_noise(img, mode='speckle', seed=None, clip=True)
    print(noisy)
    print(np.max(noisy), np.min(noisy))
    print(np.max(img), np.min(img))
    return img, noisy

def shown_next():
    img, noisy = next_image()
    axes[0].imshow(noisy - img)
    axes[1].imshow(noisy)

def press(event):
    print('press', event.key)
    axes[0].cla()
    axes[1].cla()
    if event.key == 'x':
        shown_next()
        fig.canvas.draw()


fig, axes = plt.subplots(ncols=2)
fig.canvas.mpl_connect('key_press_event', press)
shown_next()
plt.show()