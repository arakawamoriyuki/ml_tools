#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, io, csv, time, glob, datetime

import cv2
import numpy as np
from PIL import Image, ImageFile
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split


def generate_image_equals(file_path):
    cache_image_pixel = Image.open(file_path).load()
    def image_equals(path):
        image_pixel = Image.open(path).load()
        # 10pixel適当チェック
        return all(list(map(lambda i:
            image_pixel[i,i] == cache_image_pixel[i,i],
        list(range(10)))))
    return image_equals


def datasets_clean(check_dir, bad_image):
    is_bad_image = generate_image_equals(bad_image)
    for file_path in glob.glob(check_dir, recursive=True):
        bad = is_bad_image(file_path)
        print('{} {}'.format('delete' if bad else 'ok    ', file_path))
        if bad:
            os.remove(file_path)


def convert_images(src_path, dest_path, format='jpeg', size=28):
    labels = list(filter(lambda dir:
        os.path.isdir(os.path.join(src_path, dir))
    , os.listdir(src_path)))
    file_mapping = {label: glob.glob('{}/{}/**/*.*'.format(src_path, label), recursive=True) for label in labels}
    for label, file_paths in file_mapping.items():
        for file_path in file_paths:
            now = datetime.datetime.today().strftime("%Y%m%d%H%M%S%f")
            file_name = '{}.{}'.format(now, format)
            new_file_dir = '{}/{}'.format(dest_path, label)
            if not os.path.isdir(new_file_dir):
                os.makedirs(new_file_dir)
            new_file_path = '{}/{}'.format(new_file_dir, file_name)
            image = Image.open(file_path).resize((size, size))
            image.save(new_file_path, format)
            print('save   {}'.format(new_file_path))


def convert_images_edge(src_path, dest_path, format='jpeg', size=28):
    labels = list(filter(lambda dir:
        os.path.isdir(os.path.join(src_path, dir))
    , os.listdir(src_path)))
    file_mapping = {label: glob.glob('{}/{}/**/*.*'.format(src_path, label), recursive=True) for label in labels}
    for label, file_paths in file_mapping.items():
        for file_path in file_paths:
            now = datetime.datetime.today().strftime("%Y%m%d%H%M%S%f")
            file_name = '{}.{}'.format(now, format)
            new_file_dir = '{}/{}'.format(dest_path, label)
            if not os.path.isdir(new_file_dir):
                os.makedirs(new_file_dir)
            new_file_path = '{}/{}'.format(new_file_dir, file_name)
            # グレースケール
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # リサイズ
            image = cv2.resize(image, (size, size))
            # エッジ抽出
            image = convert_cv2_image_edge(image)
            cv2.imwrite(new_file_path, image)
            print('save   {}'.format(new_file_path))

def convert_cv2_image_edge(cv2_image):
    return cv2.Canny(cv2_image, 50, 110)

def convert(path, format='jpeg', size=28):
    image = Image.open(path)
    # change format
    image.save(path, format)
    # resize
    cv2_image = cv2.imread(path, cv2.IMREAD_COLOR)
    cv2_resize_image = cv2.resize(cv2_image, (size, size))
    cv2.imwrite(path, cv2_resize_image)


def image_to_vector(path, image_size=28, color=True):
    """ 画像のベクトル化
    @param
        path                画像path
        image_size          画像の1辺のpixel数
        color               カラー画像フラグ
    @return
        vectorized_image    vectorized image
    """
    color_type = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, color_type)
    img = cv2.resize(img, (image_size, image_size))
    # Unrolling Parameters
    img = img.flatten()
    # feature normalize
    vectorized_image = img.astype(np.float32) / 255.0
    return vectorized_image


def build_csv(datasets_path, csv_file_path='datasets.csv'):
    """ 画像のダウンロードとdatasets.csvの作成
    @param
        datasets_path   画像配置ディレクトリ(1階層目のフォルダ名が分類名)
        csv_file_path   出力するcsvファイルパス
    """
    csv_writer = csv.writer(
        io.open(csv_file_path, 'a', encoding='utf-8'),
        delimiter=','
    )
    labels = list(filter(lambda dir:
        os.path.isdir(os.path.join(datasets_path, dir))
    , os.listdir(datasets_path)))
    path_map = [glob.glob('{}/{}/**/*.*'.format(datasets_path, label), recursive=True) for label in labels]
    for path_tuple in zip(*path_map):
        for index, file_path in enumerate(path_tuple):
            csv_writer.writerow([file_path, labels[index]])
            print('write  {} {}'.format(labels[index], file_path))

def get_labels(csv_path):
    csv_reader = csv.reader(
        io.open(csv_path, 'r', encoding='utf-8'),
        delimiter=','
    )
    return list(set([row[1] for row in csv_reader]))

def get_datasets(csv_path, test_size=0.1, image_size=28, color=True):
    """ トレーニングとテスト用のデータセットを取得
    @param
        csv_path        データセットcsv
        test_size       データセットをテストに利用する割合
        image_size      画像の1辺のpixel数
        color           カラー画像フラグ
    @return
        x_train         トレーニングデータセット(特徴)
        x_test          テストデータセット(特徴)
        y_train         トレーニングデータセット(答えラベル)
        y_test          テストデータセット(答えラベル)
    """
    csv_reader = csv.reader(
        io.open(csv_path, 'r', encoding='utf-8'),
        delimiter=','
    )
    labels = get_labels(csv_path)
    X = []
    y = []
    for row in csv_reader:
        # ベクトル化した画像
        X.append(image_to_vector(row[0], image_size=image_size, color=color))
        # one of k方式で答えラベルを用意
        one_of_k = np.zeros(len(labels))
        one_of_k.put(labels.index(row[1]), 1)
        y.append(one_of_k)
    return train_test_split(
        np.array(X),
        np.array(y),
        test_size=test_size,
        random_state=42
    )




