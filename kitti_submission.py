#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Nghia Tran
# Edited from demo.py


"""
Detects Cars in an imaged using KittiBox.

Input:
    test_path: path to folder contain images to predict.
    logdir: Log directory for trained model
Output:
    A folder that contains txt files for bounding boxes in provided images

Usage:
python kittisubmission.py test_path logdir
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys

# configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf
import time
import argparse
from  evals.kitti_eval import write_rects

sys.path.insert(1, 'incl')
from utils import train_utils as kittibox_utils

try:
    # Check whether setup was done correctly
    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


parser = argparse.ArgumentParser(description='Create summsion for Kitti')
parser.add_argument('test_path', type=str, help='Path to test folder.')
parser.add_argument('logdir', type=str, help='Path to logdir.')
parser.add_argument('--save', '-s', type=str, default='.', help='Save directory.')


def main():
    tv_utils.set_gpus_to_use()
    args = parser.parse_args()

    logdir = args.logdir
    if not os.path.isdir(args.save):
        logging.error('--save flag must be a directory.')

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32, shape=(hypes["image_height"], hypes["image_width"], 3))
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    test_path = args.test_path
    image_names = os.listdir(test_path).sort()
    start = time.time()
    for i, image_name in enumerate(image_names):
        input_image = os.path.join(test_path, image_name)

        # Load and resize input image
        image = scp.misc.imread(input_image)
        image = scp.misc.imresize(image,
                                  (hypes["image_height"], hypes["image_width"]),
                                  interp='cubic')
        feed = {image_pl: image}

        # Run KittiBox model on image
        pred_boxes = prediction['pred_boxes_new']
        pred_confidences = prediction['pred_confidences']
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
                                                         pred_confidences],
                                                        feed_dict=feed)

        # Apply non-maximal suppression
        # and draw predictions on the image
        threshold = 0.5
        output_image, rects = kittibox_utils.add_rectangles(
            hypes, [image], np_pred_confidences,
            np_pred_boxes, show_removed=False,
            use_stitching=True, rnn_len=1,
            min_conf=threshold, tau=hypes['tau'], color_acc=(0, 255, 0))

        test_file_name = image_name.split('.')[0] + '.txt'
        test_file = os.path.join(args.save, test_file_name)

        write_rects(rects, test_file)

    time_taken = time.time() - start
    logging.info('Number of images: %d' % len(image_names))
    logging.info('Time takes: %.2f s' % (time_taken))
    logging.info('Frequency: %.2f fps' % (len(image_names) / time_taken))

if __name__ == '__main__':
    main()