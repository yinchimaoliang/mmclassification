from typing import DefaultDict

import numpy as np

from .builder import DATASETS
from .imagenet import ImageNet, find_folders, get_samples
from .pipelines import Compose


@DATASETS.register_module()
class PathologyDataset(ImageNet):

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        super(PathologyDataset, self).__init__(data_prefix, pipeline, classes,
                                               ann_file, test_mode)

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().split(' ') for x in f.readlines()]
        elif isinstance(self.ann_file, list):
            label_samples_dict = DefaultDict(list)
            for ann_file in self.ann_file:
                with open(ann_file, 'r') as f:
                    for line in f.readlines():
                        filename, label = line.split(' ')
                        label_samples_dict[filename].append(int(label[:-1]))
                samples = [[key, value]
                           for key, value in label_samples_dict.items()]

        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples
        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
