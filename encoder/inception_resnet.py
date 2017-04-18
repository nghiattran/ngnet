from __future__ import print_function

import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from nets.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2


def checkpoint_fn(version=1, layers=50):
    return 'resnet_v%d_%d.ckpt' % (version, layers)

def inference(hypes, images, train=True):
    """
    Build ResNet encoder

    :param hypes:
    :param images:
    :param train:
    :return:
    """
    is_training = tf.convert_to_tensor(train,
                                       dtype='bool',
                                       name='is_training')

    num_classes = 2
    # images = tf.placeholder(tf.float32, shape=(3, 224, 224, 3))
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logit, endpoints = inception_resnet_v2(images,
                                               num_classes=num_classes,
                                               is_training=is_training)
    for key in endpoints:
        print(key, endpoints[key].get_shape().as_list())
    print(logit.get_shape().as_list())
    exit(0)
    return {
        'early_feat': endpoints['Mixed_5b'],
        'deep_feat': endpoints['Conv2d_4a_3x3']
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

        layers = hypes['arch']['layers']

        filename = checkpoint_fn(layers=layers)

        if 'TV_DIR_DATA' in os.environ:
            filename = os.path.join(os.environ['TV_DIR_DATA'], 'weights',
                                    "tensorflow_resnet", filename)
        else:
            filename = os.path.join('DATA', 'weights', "tensorflow_resnet",
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