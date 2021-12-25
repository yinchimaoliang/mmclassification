# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path

import mmcv
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Glomerulus(BaseDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
    """  # noqa: E501

    def load_annotations(self):
        labels = mmcv.load(self.ann_file)
        data_infos = list()
        for img_name, label in labels.items():
            img = mmcv.imread(
                os.path.join(self.data_prefix, 'images', img_name))
            gt_label = self.CLASSES.index(label)
            info = {'img': img, 'gt_label': np.array(gt_label)}
            data_infos.append(info)
        return data_infos
