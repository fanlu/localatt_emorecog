
from sklearn.utils.class_weight import compute_class_weight
import torch as tc
import pickle as pk
import numpy as np

def get_class_weight(dataset_utt_lab_path, device):
    with open(dataset_utt_lab_path) as f:
        y = [int(l.split()[1]) for l in f.readlines()]

    cls = list(set(y))
    y = np.int_(np.array(y))

    cls_wgt = compute_class_weight('balanced', cls, y)
    return tc.FloatTensor(cls_wgt).to(device)


def get_class_weight(dataset_utt_lab_path):
    # classes = {'f': 0, 'm': 1}
    with open(dataset_utt_lab_path) as f:
        y = [l.strip().split()[1] for l in f.readlines()]
        classes = np.sort(np.unique(y)).tolist()
        print(classes)
        y = [classes.index(x) for x in y]

    cls = list(set(y))
    print(cls)
    y = np.int_(np.array(y))

    cls_wgt = compute_class_weight('balanced', cls, y)
    return cls_wgt, classes


if __name__ == "__main__":
    y = ['ang', 'hap', 'neu', 'sad']
    print(y.index('hap'))
    print(get_class_weight("localatt_emorecog/data/train_16k_aug/utt2spk"))