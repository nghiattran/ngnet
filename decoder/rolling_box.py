#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random
import tensorflow as tf

from utils import train_utils
from utils import data_utils


def ROI_layer(input, output_shape):
    assert len(output_shape) == 2

    ih, iw = input.get_shape().as_list()[1:3]
    h, w = output_shape

    assert ih >= h and iw >= w

    ksize = [1, int(ih/h), int(iw/w), 1]
    strides = ksize
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='VALID')

def conv_ROI_layer(name, input, output_shape):
    assert len(output_shape) == 3
    assert len(input.get_shape().as_list()) == 4

    ih, iw, ic = input.get_shape().as_list()[1:]
    h, w, c = output_shape

    assert ih >= h and iw >= w

    filter = tf.get_variable('%s_conv_roi' % (name), shape=(int(ih/h), int(iw/w), ic, c))
    strides = (1, int(ih/h), int(iw/w), 1)

    return tf.nn.conv2d(input, filter=filter, strides=strides, padding='VALID')

def decoder(hypes, logits, train):
    """Apply decoder to the logits.

    Computation which decode CNN boxes.
    The output can be interpreted as bounding Boxes.


    Args:
      logits: Logits tensor, output von encoder

    Return:
      decoded_logits: values which can be interpreted as bounding boxes
    """
    batch_size = hypes['batch_size']
    hypes['solver']['batch_size'] = batch_size
    if not train:
        hypes['batch_size'] = 1

    dlogits = {}

    initializer = tf.contrib.layers.xavier_initializer()
    l2_reg = tf.contrib.layers.l2_regularizer(hypes.get('reg_strength', 1e-6))

    grid_size = (hypes['grid_height'], hypes['grid_width'])
    outer_size = grid_size[0] * grid_size[1] * hypes['batch_size']

    out_channel = 200

    with tf.variable_scope('decoder', initializer=initializer, regularizer=l2_reg):
        with tf.variable_scope('perception'):
            conv_keys = sorted(logits['convs'].keys())
            last_conv = None

            for key in conv_keys:
                if last_conv is None:
                    last_conv = ROI_layer(logits['convs'][key], grid_size)
                else:
                    last_conv = tf.concat([last_conv, ROI_layer(logits['convs'][key], grid_size)], axis=3)

            if last_conv is None:
                memory = ROI_layer(logits['deep_feat'], (hypes['grid_height'], hypes['grid_width']))
            else:
                memory = tf.concat([last_conv, logits['deep_feat']], axis=3)

            channels = memory.get_shape().as_list()[-1]
            memory_reshape = tf.reshape(memory, shape=(outer_size, channels))


            w = tf.get_variable('ip_w', shape=(channels, out_channel))
            b = tf.get_variable('ip_b', shape=(out_channel))
            perception = tf.matmul(memory_reshape, w) + b

            if train:
                perception = tf.nn.dropout(perception, 0.5)

        with tf.variable_scope('regression_head'):
            hidden_layer = 50
            box_weights_hidden = tf.get_variable('hidden_box_out', shape=(out_channel, hidden_layer))
            box_bias_hidden = tf.get_variable('hidden_box_out_bias', shape=(hidden_layer))
            box_hidden = tf.matmul(perception, box_weights_hidden) + box_bias_hidden

            if train:
                box_hidden = tf.nn.dropout(box_hidden, 0.5)

            box_weights = tf.get_variable('box_out', shape=(hidden_layer, 4))
            box_bias = tf.get_variable('box_out_bias', shape=(4))
            pred_boxes = tf.matmul(box_hidden, box_weights) + box_bias
            pred_boxes = tf.reshape(pred_boxes, shape=[outer_size, 1, 4])

        with tf.variable_scope('classifivation_head'):
            class_weights = tf.get_variable('box_out', shape=(out_channel, hypes['num_classes']))
            class_bias = tf.get_variable('box_out_bias', shape=(hypes['num_classes']))
            pred_logits = tf.matmul(perception, class_weights) + class_bias
            pred_logits = tf.reshape(pred_logits, shape=[outer_size, hypes['num_classes']])
            pred_confidences = tf.nn.softmax(pred_logits)

    dlogits['pred_boxes'] = pred_boxes
    dlogits['pred_logits'] = pred_logits
    dlogits['pred_confidences'] = pred_confidences
    hypes['batch_size'] = batch_size

    return dlogits

def loss(hypes, decoded_logits, labels):
    """Calculate the loss from the logits and the labels.

    Args:
      decoded_logits: output of decoder
      labels: Labels tensor; Output from data_input

      flags: 0 if object is present 1 otherwise
      confidences: ??
      boxes: encoding of bounding box location

    Returns:
      loss: Loss tensor of type float.
    """
    confidences, boxes, mask = labels

    pred_boxes = decoded_logits['pred_boxes']
    pred_logits = decoded_logits['pred_logits']
    pred_confidences = decoded_logits['pred_confidences']

    # grid_size = (hypes['grid_width'], hypes['grid_height'])
    outer_size = pred_boxes.get_shape().as_list()[0]

    head = hypes['solver']['head_weights']

    # Compute confidence loss
    confidences = tf.reshape(confidences, (outer_size, 1))
    true_classes = tf.reshape(tf.cast(tf.greater(confidences, 0), 'int64'),
                              [outer_size])

    pred_classes = tf.reshape(pred_logits, [outer_size, hypes['num_classes']])
    mask_r = tf.reshape(mask, [outer_size])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred_classes, labels=true_classes)

    # ignore don't care areas
    cross_entropy_sum = (tf.reduce_sum(mask_r * cross_entropy))
    confidences_loss = cross_entropy_sum / outer_size * head[0]

    true_boxes = tf.reshape(boxes, (outer_size, hypes['rnn_len'], 4))

    # box loss for background prediction needs to be zerod out
    boxes_mask = tf.reshape(tf.cast(tf.greater(confidences, 0), 'float32'), (outer_size, 1, 1))

    # danger zone
    pred_boxes_flat = tf.reshape(pred_boxes * boxes_mask, [-1, 4])
    perm_truth_flat = tf.reshape(true_boxes, [-1, 4])
    iou = train_utils.iou(train_utils.to_x1y1x2y2(pred_boxes_flat),
                          train_utils.to_x1y1x2y2(perm_truth_flat))
    boxes_loss = -tf.reduce_sum(tf.log(tf.maximum(iou, 1e-3))) / (tf.reduce_sum(boxes_mask) + 1e-6)
    loss = confidences_loss + boxes_loss

    tf.add_to_collection('loss', loss)

    reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES

    decoder_reg_loss = tf.add_n(tf.get_collection(reg_loss_col, scope='decoder'), name='decoder_reg_loss')
    weight_loss = tf.add_n(tf.get_collection(reg_loss_col, scope=''), name='reg_loss')
    encoder_reg_loss = tf.subtract(weight_loss, decoder_reg_loss, name='encoder_reg_loss')


    total_loss = weight_loss + loss

    tf.summary.scalar('/weights', weight_loss)
    tf.summary.scalar('/decoder_reg_loss', decoder_reg_loss)
    # tf.summary.scalar('/encoder_reg_loss', encoder_reg_loss)

    losses = {}
    losses['total_loss'] = total_loss
    losses['decoder_reg_loss'] = decoder_reg_loss
    losses['encoder_reg_loss'] = encoder_reg_loss
    losses['loss'] = loss
    losses['confidences_loss'] = confidences_loss
    losses['boxes_loss'] = boxes_loss
    losses['weight_loss'] = weight_loss

    return losses


def evaluation(hyp, images, labels, decoded_logits, losses, global_step):
    """
    Compute summary metrics for tensorboard
    """

    pred_confidences = decoded_logits['pred_confidences']
    pred_boxes = decoded_logits['pred_boxes']
    # Estimating Accuracy
    grid_size = hyp['grid_width'] * hyp['grid_height']
    confidences, boxes, mask = labels

    new_shape = [hyp['batch_size'], hyp['grid_height'],
                 hyp['grid_width'], hyp['num_classes']]
    pred_confidences_r = tf.reshape(pred_confidences, new_shape)
    # Set up summary operations for tensorboard
    a = tf.equal(tf.cast(confidences, 'int64'),
                 tf.argmax(pred_confidences_r, 3))

    accuracy = tf.reduce_mean(tf.cast(a, 'float32'), name='/accuracy')

    eval_list = []
    eval_list.append(('Acc.', accuracy))
    eval_list.append(('Conf', losses['confidences_loss']))
    eval_list.append(('Box', losses['boxes_loss']))
    eval_list.append(('Weight', losses['weight_loss']))

    # Log Images
    # show ground truth to verify labels are correct
    pred_confidences_r = tf.reshape(
        pred_confidences,
        [hyp['batch_size'], grid_size, hyp['rnn_len'], hyp['num_classes']])

    # show predictions to visualize training progress
    pred_boxes_r = tf.reshape(
        pred_boxes, [hyp['batch_size'], grid_size, hyp['rnn_len'],
                     4])
    test_pred_confidences = pred_confidences_r[0, :, :, :]
    test_pred_boxes = pred_boxes_r[0, :, :, :]

    def log_image(np_img, np_confidences, np_boxes, np_global_step,
                  pred_or_true):

        if pred_or_true == 'pred':
            plot_image = train_utils.add_rectangles(
                hyp, np_img, np_confidences, np_boxes, use_stitching=True,
                rnn_len=hyp['rnn_len'])[0]
        else:
            np_mask = np_boxes
            plot_image = data_utils.draw_encoded(
                np_img[0], np_confidences[0], mask=np_mask[0], cell_size=32)

        num_images = 10

        filename = '%s_%s.jpg' % \
            ((np_global_step // hyp['logging']['write_iter'])
                % num_images, pred_or_true)
        img_path = os.path.join(hyp['dirs']['output_dir'], filename)

        scp.misc.imsave(img_path, plot_image)
        return plot_image

    pred_log_img = tf.py_func(log_image,
                              [images, test_pred_confidences,
                               test_pred_boxes, global_step, 'pred'],
                              [tf.float32])

    true_log_img = tf.py_func(log_image,
                              [images, confidences,
                               mask, global_step, 'true'],
                              [tf.uint8])
    tf.summary.image('/pred_boxes', tf.stack(pred_log_img))
    tf.summary.image('/true_boxes', tf.stack(true_log_img))
    return eval_list
