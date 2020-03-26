import glob
a = open("/mnt/cephfs2/asr/users/fanlu/kaldi2/egs/m8/data/train/spk2gender").readlines()
b = open("/mnt/cephfs2/asr/users/fanlu/kaldi2/egs/m8/data/train/wav.scp").readlines()
d = open("/mnt/cephfs2/asr/users/fanlu/kaldi2/egs/m8/data/train/utt2dur").readlines()
c = open("/mnt/cephfs2/asr/users/fanlu/localatt_emorecog/kefu/wav_cat_train.list", "w")
cls = {'f':'anger', 'm':'neutral'}
classes = {'f': 0, 'm': 1}
for l in zip(a,b,d):
    label = l[0].strip().split()[1]
    wav = l[1].strip().split()[1]
    dur = float(l[2].strip().split()[1])
    if dur > 16:
        continue
    c.write("%s %s\n" % (wav, classes.get(label)))
c.close()

