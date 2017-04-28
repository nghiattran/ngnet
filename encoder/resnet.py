from __future__ import print_function

import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
from tensorflow.contrib.slim.python.slim.nets import resnet_v1


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

    layers = hypes['arch']['layers']
    deep_feat = hypes['arch'].get('deep_feat', 'block4')
    early_feat = hypes['arch'].get('early_feat', 'block1')

    blocks = ['block1', 'block2', 'block3', 'block4']

    assert early_feat in blocks
    assert deep_feat in blocks[1:]

    if layers == 50:
        resnet = resnet_v1.resnet_v1_50
    elif layers == 101:
        resnet = resnet_v1.resnet_v1_101
    elif layers == 152:
        resnet = resnet_v1.resnet_v1_152
    else:
        logging.error('Resnet only has 50, 101, or 152 layers. Got', layers)
        exit(1)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training)):
        logits, endpoints = resnet(images)

        for name in blocks:
            layer_name = 'resnet_v1_%d/%s' % (layers, name)
            tf.summary.histogram('/%s_activation' % name, endpoints[layer_name])
            tf.summary.scalar('/%s_sparsity' % name, tf.nn.zero_fraction(endpoints[layer_name]))

    if train:
        restore = tf.global_variables()
        hypes['init_function'] = _initalize_variables
        hypes['restore'] = restore

    return {
        'early_feat': endpoints['resnet_v1_%d/%s' % (layers, early_feat)],
        'deep_feat': endpoints['resnet_v1_%d/%s' % (layers, deep_feat)]
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