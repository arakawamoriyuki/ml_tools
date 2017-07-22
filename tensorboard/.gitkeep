# coding: utf-8

import os, sys

# loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../lib')

from config import DATASETS_CSV, MODEL_PATH, TENSORBOARD_PATH, BAD_FILE, IMAGE_SIZE, CHANNELS, TEST_SIZE
from datasets import get_datasets, get_labels
from classify_image import run_training


def main():
    # データセットの取得
    datasets = get_datasets(DATASETS_CSV, test_size=TEST_SIZE, image_size=IMAGE_SIZE)
    train_data, _, _, _ = datasets

    # TODO
    # 29 0.9961215257644653
    # 30回目で過学習により5割減

    # 30回に分けてトレーニング実行
    max_steps = 25
    run_training(
        datasets,
        tensorboard_path=TENSORBOARD_PATH,
        checkpoint_path=MODEL_PATH,
        # 分類数
        num_classes=len(get_labels(DATASETS_CSV)),
        # 画像サイズ
        image_size=IMAGE_SIZE,
        # ピクセルのベクトル数 3=カラー,1=モノクロ
        channel=CHANNELS,
        # 学習実行回数
        max_steps=max_steps,
        # 1エポックで学習するデータサイズ
        batch_size=int(len(train_data) / max_steps),
        # 学習率
        learning_rate=1e-4
    )


if __name__ == "__main__":
    main()