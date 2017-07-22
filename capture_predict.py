# coding: utf-8

# python capture_predict.py -c 2

import os, sys

# loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../lib')

import argparse
import numpy as np

from config import IMAGE_SIZE, MODEL_PATH, CHANNELS
from capture import generate_frame
from classify_image import generate_run_inference_on_vector

def main(classes=10):
    run_inference_on_vector = generate_run_inference_on_vector(
        list(range(classes)),
        image_size=IMAGE_SIZE,
        checkpoint_path=MODEL_PATH,
        channel=CHANNELS
    )

    for frame, show_frame in generate_frame():

        # unrolling parameters and feature normalize
        image_flatten = frame.flatten()
        vectorized_image = image_flatten.astype(np.float32) / 255.0

        results = run_inference_on_vector(vectorized_image)

        results = sorted(results, key=lambda result: result['score'])

        predict_text = '{} {}'.format(results[0]['label'], results[0]['score'])
        show_frame(
            frame,
            zoom=4,
            text=predict_text,
            color=(255,255,255)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classes', default='10', help='class count')
    args = parser.parse_args()
    main(classes=int(args.classes))