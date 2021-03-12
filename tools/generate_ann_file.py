import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Generate ann file')
    parser.add_argument(
        '--data-root',
        default='./data/karyotype_polarity',
        help='Root path of data')
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--valid-ratio', type=float, default=0.2)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    folders = os.listdir(args.data_root)
    annos = []
    for i, folder in enumerate(folders):
        file_names = os.listdir(os.path.join(args.data_root, folder))
        annos += [[os.path.join(folder, name), i] for name in file_names]
    np.random.shuffle(annos)
    train_annos = annos[:int(len(annos) * args.train_ratio)]
    valid_annos = annos[int(len(annos) * args.train_ratio
                            ):int(len(annos) * args.train_ratio) +
                        int(len(annos) * args.valid_ratio)]
    test_annos = annos[int(len(annos) * args.train_ratio) +
                       int(len(annos) * args.valid_ratio):]
    with open(os.path.join(args.data_root, 'train.txt'), 'w') as f:
        for path, label in train_annos:
            f.write(f'{path} {label}\n')

    with open(os.path.join(args.data_root, 'valid.txt'), 'w') as f:
        for path, label in valid_annos:
            f.write(f'{path} {label}\n')

    with open(os.path.join(args.data_root, 'test.txt'), 'w') as f:
        for path, label in test_annos:
            f.write(f'{path} {label}\n')


if __name__ == '__main__':
    main()
