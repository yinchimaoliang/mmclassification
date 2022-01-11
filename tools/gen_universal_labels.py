import mmcv
import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate universal labels.')
    parser.add_argument('--origin-path', help='Path of the original data.')
    parser.add_argument('--target-path', help='Target path to save universal label.')
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    origin_path = args.origin_path
    labels = mmcv.load(os.path.join(origin_path, 'labels.pkl'))
    target_path = args.target_path
    
    split_categories = list()
    for _, category in labels.items():
        split_categories.extend(category.split('+'))
    processed_categories = list(set(split_categories))
    for processed_category in processed_categories:
        mmcv.mkdir_or_exist(os.path.join(target_path, processed_category, 'images'))
    processed_labels = [dict() for _ in range(len(processed_categories))]
    for img_name, label in labels.items():
        sub_labels = label.split('+')
        for sub_label in sub_labels:
            shutil.copyfile(os.path.join(origin_path, 'images', img_name), os.path.join(os.path.join(target_path, sub_label, 'images', img_name)))
            label_idx = processed_categories.index(sub_label)
            processed_labels[label_idx][img_name] = sub_label
    for processed_category, processed_label in zip(processed_categories, processed_labels):
        mmcv.dump(processed_label, os.path.join(target_path, processed_category, 'labels.pkl'))
        
            
    
if __name__ == '__main__':
    main()