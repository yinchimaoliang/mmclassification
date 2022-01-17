import argparse
import os

import mmcv
import numpy as np
from tqdm import tqdm

classes = ['C', 'E', 'GS', 'M', 'SS', 'GM', 'N']


def parse_args():
    parser = argparse.ArgumentParser(description='Generate instances.')
    parser.add_argument('--pkl-path', help='Path of the json file.')
    parser.add_argument('--img-path', help='Path of images.')
    parser.add_argument('--save-path', help='Path to save the images.')
    parser.add_argument(
        '--option', nargs='+', default='save', help='Show or save.')
    args = parser.parse_args()
    return args


def _get_sub_labels(num_image, coco_data, img_path, save_path):
    cats = coco_data.cats
    cat_names = [cat['name'] for cat in cats.values()]
    catIds = []
    mmcv.mkdir_or_exist(os.path.join(save_path, 'images'))
    list_imgIds = coco_data.getImgIds(catIds=catIds)
    img = coco_data.loadImgs(list_imgIds[num_image])[0]
    image = mmcv.imread(os.path.join(img_path, img['file_name']))
    img_annIds = coco_data.getAnnIds(
        imgIds=img['id'], catIds=catIds, iscrowd=None)
    img_anns = coco_data.loadAnns(img_annIds)
    bboxes = np.array([img_ann['bbox'] for img_ann in img_anns])
    if bboxes.shape[0] == 0:
        return list(), image, img['file_name']
    # Convert xywh to xyxy.
    bboxes[:, 2:] += bboxes[:, :2]
    labels = np.array([img_ann['category_id'] for img_ann in img_anns])
    assert bboxes.shape[0] == labels.shape[0]
    sub_labels = list()
    for label in labels:
        label_name = cat_names[label]
        sub_labels.extend(label_name.split('+'))
    return list(set(sub_labels)), image, img['file_name']


def main():
    args = parse_args()
    pkl_path = args.pkl_path
    img_path = args.img_path
    save_path = args.save_path
    # TODO: Support save and show mode.
    # option = args.option
    labels = mmcv.load(pkl_path)
    mmcv.mkdir_or_exist(save_path)
    pkls = [dict() for _ in range(len(classes))]
    images = [list() for _ in range(len(classes))]
    for class_name in classes:
        mmcv.mkdir_or_exist(os.path.join(save_path, class_name, 'images'))
    with tqdm(total=len(labels)) as pbar:
        for img_name, label in labels.items():
            sub_labels = label.split('+')
            image = mmcv.imread(os.path.join(img_path, img_name))
            for sub_label in sub_labels:
                mmcv.imwrite(
                    image,
                    os.path.join(save_path, sub_label, 'images', img_name))
                label_idx = classes.index(sub_label)
                images[label_idx].append(image)
                pkls[label_idx][img_name] = sub_label
            for class_name in classes:
                if class_name not in sub_labels:
                    label_idx = classes.index(class_name)
                    mmcv.imwrite(
                        image,
                        os.path.join(save_path, class_name, 'images',
                                     img_name))
                    pkls[label_idx][img_name] = 'background'
            pbar.update()
    for pkl, class_name in zip(pkls, classes):
        mmcv.dump(pkl, os.path.join(save_path, class_name, 'labels.pkl'))


if __name__ == '__main__':
    main()
