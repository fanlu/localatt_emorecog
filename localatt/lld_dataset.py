# xxx egemaps_dataset.py
import os
import pickle as pk
import numpy as np
import torch as tc
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from toolz import curry
from fileutils.htk import readHtk
import kaldiio
import scipy.io.wavfile as wav
from python_speech_features import mfcc, logfbank


class lld_dataset(Dataset):

    def __init__(self, utt_lld, utt_lab_path, cls_wgt, device):
        import pdb
        pdb.set_trace()
        with open(utt_lab_path) as f:
            utt_lab = [l.strip().split() for l in f.readlines()]
            # utt label

        self.samples = [0] * len(utt_lab)

        for i, pair in enumerate(utt_lab):
            utt, label = pair
            label = int(label)

            lld = utt_lld[utt]

            self.samples[i] = (
                Variable(tc.FloatTensor(lld)).to(device),  # input
                len(lld),  # len
                Variable(tc.LongTensor(np.array([label]))).to(device),  # label
                cls_wgt[label])  # class weight

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class lld_dataset2(Dataset):
    def __init__(self, utt_lab_path, cls_wgt, device):
        self.samples = []
        self.device = device
        self.cls_wgt = cls_wgt
        with open(utt_lab_path) as f:
            utt_lab = [l.strip().split() for l in f.readlines()]
        for i, pair in enumerate(utt_lab):
            utt, label = pair
            self.samples.append((utt, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt, label = self.samples[idx]
        (rate, sig) = wav.read(utt)
        mfcc_feat = mfcc(sig, rate, numcep=32)
        return (tc.FloatTensor(mfcc_feat).to(self.device), len(mfcc_feat), tc.LongTensor(np.array([int(label)])).to(self.device), self.cls_wgt[int(label)])


@curry
def lld_collate_fn(data, device):

    data.sort(key=lambda x: x[1], reverse=True)
    llds, lens, labels, wgts = zip(*data)

    padded_llds = tc.zeros(
        [len(llds), max(lens), llds[0].size()[1]]).to(device)
    for i, lld in enumerate(llds):
        padded_llds[i, 0:lens[i], :] = lld

    return ((padded_llds, lens),
            Variable(tc.stack(labels), requires_grad=False).view(-1),
            wgts)


class lld_dataset3(Dataset):
    def __init__(self, utt_lab_path, cls_wgt, classes, label_path=None, normalize=False, feat_type="opensmile"):
        self.samples = []
        self.cls_wgt = cls_wgt
        self.normalize = normalize
        self.feat_type = feat_type
        self.open_smile = "/mnt/cephfs2/asr/users/fanlu/opensmile-2.3.0/"
        # with open(utt_lab_path) as f:
        #    utt_lab = [l.strip().split() for l in f.readlines()]
        # self.classes = {'f': 0, 'm': 1}
        self.classes = classes
        #import pdb;pdb.set_trace()
        with open(utt_lab_path) as f, open(label_path) as f1:
            for l in zip(f.readlines(), f1.readlines()):
                self.samples.append((l[0].strip().split()[0], l[0].strip().split()[
                                    1], self.classes.index(l[1].strip().split()[1])))
        # for i, pair in enumerate(utt_lab):
        #    utt, label = pair
        #    self.samples.append((utt, label))
        print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, utt, label = self.samples[idx]
        if self.feat_type == "fbank":
            (rate, sig) = wav.read(utt)
            feat = logfbank(sig, rate)
        elif self.feat_type == "mfcc":
            (rate, sig) = wav.read(utt)
            feat = mfcc(sig, rate, numcep=32)
        elif self.feat_type == "opensmile":
            tmp_htk = "tmpdir/%s.htk" % os.path.basename(utt)
            smile = "%s/bin/linux_x64_standalone_libstdc6/SMILExtract -C %s/config/localatt_lld.conf -I %s -O %s" % (
                self.open_smile, self.open_smile, utt, tmp_htk)
            if not os.path.exists(tmp_htk):
                os.system(smile)
            feat = readHtk(tmp_htk)
        elif self.feat_type == "kaldi_mfcc":
            feat = kaldiio.load_mat(utt)
        if self.normalize:
            feat = feat - feat.mean(axis=0)
        return (feat, len(feat), np.array([int(label)]), self.cls_wgt[int(label)], key)


def lld_collate_fn_2(data):

    data.sort(key=lambda x: x[1], reverse=True)
    llds, lens, labels, wgts, keys = zip(*data)
    #import pdb;pdb.set_trace()
    padded_llds = tc.zeros([len(llds), max(lens), llds[0].shape[1]])
    for i, lld in enumerate(llds):
        padded_llds[i, 0:lens[i], :] = tc.from_numpy(lld)
    return ((padded_llds, tc.tensor(lens), keys),
            tc.from_numpy(np.array(labels)).view(-1), wgts)


if __name__ == "__main__":
    # open_smile = "/mnt/cfs2/asr/users/fanlu/opensmile-2.3.0/"
    # tmp_htk = "tmpdir/%s.htk" % os.path.basename(utt)
    # smile = "%s/bin/linux_x64_standalone_libstdc6/SMILExtract -C %s/config/localatt_lld.conf -I %s -O %s" % (
    #     open_smile, open_smile, utt, tmp_htk)
    # if not os.path.exists(tmp_htk):
    #     os.system(smile)
    feat = readHtk("/mnt/cfs1_alias1/asr/users/fanlu/task/localatt_emorecog/tmpdir/000010172.WAV.htk")
    print(feat.shape)
