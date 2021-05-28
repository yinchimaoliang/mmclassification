import argparse
import os

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--ann-dir', help='segmentation annotation path')
    parser.add_argument('--img-dir', help='image path')
    parser.add_argument(
        '--tar-path', help='target path for classification path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    img_dir = args.img_dir
    ann_dir = args.ann_dir
    tar_path = args.tar_path

    names = os.listdir(ann_dir)
    anns = []

    for i, name in enumerate(names):
        ann = mmcv.imread(os.path.join(ann_dir, name), 0)
        anns.append(ann.max())
        print(f'{name} finished')

    with open(os.path.join(tar_path), 'w') as f:
        for i, ann in enumerate(anns):
            f.write(f'{os.path.join(img_dir, names[i])} {ann}\n')


if __name__ == '__main__':
    main()
