from __future__ import print_function

import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
from tensorflow.contrib.slim.python.slim.nets import inception_v1, inception_v2, inception_v3
from nets import inception_v4

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

    version = hypes['arch']['version']

    if version == 1:
        inception = inception_v1.inception_v1
        argscope = inception_v1.inception_v1_arg_scope
    elif version == 2:
        inception = inception_v2.inception_v2
        argscope = inception_v2.inception_v2_arg_scope
    elif version == 3:
        inception = inception_v3.inception_v3
        argscope = inception_v3.inception_v3_arg_scope
    elif version == 4:
        inception = inception_v4.inception_v4
        argscope = inception_v4.inception_v4_arg_scope
    else:
        logging.error('Inception version only has values of 1, 2, or 3 layers. Got', version)
        exit(1)

    with slim.arg_scope(argscope()):
        logit, endpoints = inception(images,
                                     num_classes=2,
                                     is_training=is_training)

    # for key in endpoints:
    #     print(key, endpoints[key].get_shape())
    # exit()

    # Sanity check
    layers = endpoints.keys()
    layers_dict = dict(zip(layers, range(len(layers))))
    assert 'early_feat' in hypes['arch']
    assert 'deep_feat' in hypes['arch']
    assert layers_dict[hypes['arch']['early_feat']] < layers_dict[hypes['arch']['deep_feat']]

    if train:
        restore = tf.global_variables()
        hypes['init_function'] = _initalize_variables
        hypes['restore'] = restore

    return {
        'early_feat': endpoints[hypes['arch']['early_feat']],
        'deep_feat': endpoints[hypes['arch']['deep_feat']]
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