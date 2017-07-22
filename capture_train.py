# coding: utf-8

# python capture_train.py -c 2 -i 2

import os, sys

# loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../lib')

import argparse, cv2
import numpy as np

from capture import generate_frame
from classify_image import generate_run_training
from config import MODEL_PATH, IMAGE_SIZE, CHANNELS


def main(classes, index):
    run_train, save = generate_run_training(
        checkpoint_path=MODEL_PATH,
        num_classes=classes,
        image_size=IMAGE_SIZE,
        channel=CHANNELS,
        learning_rate=1e-4
    )

    # label one_of_k
    one_of_k = np.zeros(classes)
    one_of_k[index-1] = 1

    # run video capture
    for frame, show_frame in generate_frame(image_size=IMAGE_SIZE, destroy_callback=save):
        show_frame(frame, zoom=4)

        # unrolling parameters and feature normalize
        image_flatten = frame.flatten()
        vectorized_image = image_flatten.astype(np.float32) / 255.0

        # training
        run_train(np.array([vectorized_image]), np.array([one_of_k]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classes', default='10', help='class count')
    parser.add_argument('-i', '--index', default='1', help='training class index')
    args = parser.parse_args()
    main(int(args.classes), int(args.index))

#