#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import backend as K

# TODO: model保存、更新
# TODO: tensorboard連携

def model(input, num_classes=10):
    with tf.variable_scope('mlp_model'):
        x = keras.layers.Dense(units=512, activation='relu')(input)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(units=512, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        y_pred = keras.layers.Dense(units=num_classes, activation='softmax')(x)
    return y_pred


def run_training(datasets, tensorboard_path='/tmp/data', checkpoint_path='./models', num_classes=10, image_size=28, max_steps=10, batch_size=100, learning_rate=1e-4):
    """ トレーニングの実行
    @param
        datasets        データセットタプル(train_images, test_images, train_labels, test_labels)
        num_classes     分類数
        image_size      画像の1辺のpixel数
        max_steps       トレーニング実行回数
        batch_size      1回のトレーニングに使用する画像枚数
        learning_rate   学習率
    """
    # datasets.csvからデータセットを取得
    # - vectrizeされた画像(特徴)
    # - one of k方式のラベル(答え)
    train_images, test_images, train_labels, test_labels = datasets

    # tensorflow placeholders
    x = tf.placeholder(tf.float32, [None, image_size*image_size*3])
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    # define TF graph
    y_pred = model(x, num_classes=num_classes)
    loss = tf.losses.softmax_cross_entropy(y_, y_pred)
    train_step = tf.train.AdagradOptimizer(0.05).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print('Training...')
        for step in range(max_steps):
            for i in range(int(len(train_images) / batch_size)):
                # batch_size分の画像に対して訓練の実行
                batch = batch_size * i
                # feed_dictでplaceholderに入れるデータを指定する
                train_step.run(feed_dict={
                    x: train_images[batch:batch+batch_size],
                    y_: train_labels[batch:batch+batch_size],
                    K.learning_phase(): 1
                })
                if i % 1000 == 0:
                    val_accuracy = accuracy.eval({
                        x: test_images,
                        y_: test_labels,
                        K.learning_phase(): 0
                    })
                    print('  step, accurary = %6d: %6.3f' % (i, val_accuracy))
        test_accuracy = accuracy.eval({
            x: test_images,
            y_: test_labels,
            K.learning_phase(): 0
        })
        print('Test accuracy:', test_accuracy)





