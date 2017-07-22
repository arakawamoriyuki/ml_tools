# coding: utf-8

import os, sys

# loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../lib')

from datasets import datasets_clean, convert_images, build_csv
from config import DATASETS_SRC, DATASETS_DEST, DATASETS_CSV, BAD_FILE, IMAGE_FORMAT, IMAGE_SIZE


def main():
    if BAD_FILE:
        # BAD_FILE指定の画像を削除する
        check_dir = '{}/**/*.*'.format(DATASETS_SRC)
        datasets_clean(check_dir, BAD_FILE)

    # 画像変換
    convert_images(DATASETS_SRC, DATASETS_DEST, format=IMAGE_FORMAT, size=IMAGE_SIZE)

    # csvを作成する
    build_csv(DATASETS_DEST, csv_file_path=DATASETS_CSV)


if __name__ == "__main__":
    main()