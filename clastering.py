# coding: utf-8

import os, sys

# loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../lib')

import io, csv, shutil, ntpath

from sklearn.cluster import KMeans
from config import DATASETS_CSV

from datasets import get_datasets


X, _, _, _ = get_datasets(DATASETS_CSV, test_size=0, image_size=28)

clasters = KMeans(n_clusters=30).fit_predict(X)

lines = csv.reader(
    io.open(DATASETS_CSV, 'r', encoding='utf-8'),
    delimiter=','
)

for line, claster in zip(lines, clasters):
    save_path = 'datasets/clastering/{0:05d}'.format(claster)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    file_name = ntpath.basename(line[0])
    dest_file = '{}/{}'.format(save_path, file_name)
    print(dest_file)
    shutil.copy(line[0], dest_file)