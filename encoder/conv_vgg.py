from __future__ import print_function

import os

import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import layers
from tensorflow.contrib.slim.python.slim.nets import  vgg


def conv_vgg_16(inputs, scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with slim.variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      end_points = utils.convert_collection_to_dict(end_points_collection)
      return net, end_points

def inference(hypes, images, train=True):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        if not train:
            tf.get_variable_scope().reuse_variables()
        logits, endpoints = conv_vgg_16(images)

    if train:
        restore = tf.global_variables()
        hypes['init_function'] = _initalize_variables
        hypes['restore'] = restore
    else:
        for key in endpoints.keys():
            realkey = '/'.join(key.split('/')[2:])
            endpoints[realkey] = endpoints[key]

    return {
        'convs': {
            'conv1_2': endpoints['vgg_16/conv1/conv1_2'],
            'conv2_2': endpoints['vgg_16/conv2/conv2_2'],
            'conv3_3': endpoints['vgg_16/conv3/conv3_3'],
            'conv4_3': endpoints['vgg_16/conv4/conv4_3']
        },
        "early_feat": endpoints['vgg_16/conv4/conv4_3'],
        'deep_feat': logits
    }


def _initalize_variables(hypes):
    if hypes['load_pretrained']:
        logging.info("Pretrained weights are loaded.")
        logging.info("The model is fine-tuned from previous training.")
        restore = hypes['restore']
        init = tf.global_variables_initializer()
        sess = tf.get_default_session()
        sess.run(init)

        saver = tf.train.Saver(var_list=restore)

        filename = 'vgg_16.ckpt'

        if 'TV_DIR_DATA' in os.environ:
            filename = os.path.join(os.environ['TV_DIR_DATA'], 'weights',
                                    "tensorflow_resnet", filename)
        else:
            filename = os.path.join('DATA', 'weights', "tensorflow_vgg",
                                    filename)

        if not os.path.exists(filename):
            logging.error("File not found: {}".format(filename))
            logging.error("Please download weights from here: {}"
                          .format('network_url'))
            exit(1)

        logging.info("Loading weights from disk.")
        saver.restore(sess, filename)
    else:
        logging.info("Random initialization performed.")
        sess = tf.get_default_session()
        init = tf.global_variables_initializer()
        sess.run(init)