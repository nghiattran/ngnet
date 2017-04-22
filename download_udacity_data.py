"""
Download Udacity Autti and crowdai
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import shutil
import subprocess
import sys
import os
import tarfile
from six.moves import urllib

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

TRAINING_FOLDER = 'training'
IMAGE_FOLDER = 'image_2'
LABEL_FOLDER = 'label_2'

datasets = {
    'crowdai': {
        'url': 'http://bit.ly/udacity-annoations-crowdai',
        'tar_file': 'object-detection-crowdai.tar.gz',
        'extracted_dirname': 'object-detection-crowdai'
    },
    'autti': {
        'url': 'http://bit.ly/udacity-annotations-autti',
        'tar_file': 'object_dataset.tar.gz',
        'extracted_dirname': 'object-dataset'
    }
}

def get_pathes():
    """
    Get location of `data_dir` and `run_dir'.

    Defaut is ./DATA and ./RUNS.
    Alternativly they can be set by the environoment variabels
    'TV_DIR_DATA' and 'TV_DIR_RUNS'.
    """

    if 'TV_DIR_DATA' in os.environ:
        data_dir = os.path.join(['hypes'], os.environ['TV_DIR_DATA'])
    else:
        data_dir = "DATA"

    if 'TV_DIR_RUNS' in os.environ:
        run_dir = os.path.join(['hypes'], os.environ['TV_DIR_DATA'])
    else:
        run_dir = "RUNS"

    return data_dir, run_dir


def download(url, dest):
    filename = url.split('/')[-1]

    logging.info("Download URL: {}".format(url))
    logging.info("Download FILE: {}".format(dest))

    def _progress(count, block_size, total_size):
                prog = float(count * block_size) / float(total_size) * 100.0
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename, prog))
                sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()
    return filepath

def handle_data(udacity_dec_dir, type):
    dataset = datasets[type.lower()]
    url = dataset['url']
    data_tar = dataset['tar_file']
    extracted_dirname = dataset['extracted_dirname']

    # If data is not downloaded yet, download
    data_path = os.path.join(udacity_dec_dir, data_tar)
    if os.path.exists(data_path):
        logging.info("Udacity %s compressed file FOUND at %s." % (type, data_path))
    else:
        logging.info("Downloading Udacity %s Data." % type)
        data_path = download(url, os.path.join(udacity_dec_dir, data_tar))
        logging.info("Done extracting Udacity %s Data." % type)

    # If data is not extracted yet, extract
    data_extracted_folder = os.path.join(udacity_dec_dir, extracted_dirname)
    if os.path.exists(data_extracted_folder):
        logging.info("Udacity %s extracted data FOUND at %s." % (type, data_extracted_folder))
    else:
        logging.info("Extracting Udacity %s Data." % type)
        tarfile.open(data_path, "r").extractall(udacity_dec_dir)

    # Convert Autti data to Kitti
    converted_path = os.path.join(udacity_dec_dir, type.lower())
    try:
        logging.info("Converting Udacity %s Data." % type)
        data_tar = os.path.join(udacity_dec_dir, extracted_dirname)
        convert_to_kitti(data_tar, converted_path, 'udacity-%s' % type.lower())
        logging.info("Done converting Udacity %s Data." % type)
    except Exception as e:
        logging.warning("Failed to convert Udacity %s to Kitti." % type)
        logging.warning(
            "Make sure you have all submodules installed by running: git submodule update --init --recursive")
        exit(1)

    src = os.path.join(converted_path, TRAINING_FOLDER, IMAGE_FOLDER)
    for file in os.listdir(src):
        image_src = os.path.join(converted_path, TRAINING_FOLDER, IMAGE_FOLDER, file)
        image_dst = os.path.join(udacity_dec_dir, TRAINING_FOLDER, IMAGE_FOLDER, file)
        shutil.move(src=image_src,
                    dst=image_dst)

    new_data = []
    with open(os.path.join(converted_path, 'train.txt'), 'r') as f:
        data = f.readlines()
        for id in data:
            id = id[:-1]
            entry = '%s %s' % (os.path.join(TRAINING_FOLDER, IMAGE_FOLDER, id + '.jpg'),
                               os.path.join(TRAINING_FOLDER, LABEL_FOLDER, id + '.txt'))
            new_data.append(entry)

    # Remove data to save space
    shutil.rmtree(converted_path)
    shutil.rmtree(data_extracted_folder)

    return new_data

def convert_to_kitti(from_path, to_path, type):
    subprocess.check_call([
        'python', 'submodules/vod-converter/vod_converter/main.py',
        '--from', type,
        '--from-path', from_path,
        '--to', 'kitti',
        '--to-path', to_path
    ])
    return to_path

if __name__ == '__main__':
    data_dir, run_dir = get_pathes()

    # Create folder structure for dataset
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    udacity_dec_dir = os.path.join(data_dir, 'Udacity')
    if not os.path.exists(udacity_dec_dir):
        os.makedirs(udacity_dec_dir)

    training_folder = os.path.join(udacity_dec_dir, TRAINING_FOLDER)

    image_folder = os.path.join(training_folder, IMAGE_FOLDER)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    label_folder = os.path.join(training_folder, LABEL_FOLDER)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    # Get data
    crowdai_data = handle_data(udacity_dec_dir, type='Crowdai')
    autti_data = handle_data(udacity_dec_dir, type='Autti')

    joint_data = crowdai_data + autti_data
    split = int(len(joint_data) * 0.05)

    with open(os.path.join(udacity_dec_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(joint_data[split:]))
    logging.info("Done writing train.txt")

    with open(os.path.join(udacity_dec_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(joint_data[:split]))
    logging.info("Done writing val.txt")