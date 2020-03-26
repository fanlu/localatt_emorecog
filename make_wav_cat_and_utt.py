import os
f2 = open("iemocap/wav_cat.list", "w")
f3 = open("iemocap/utt.list", "w")
classes = {'ang': 'A', 'hap': 'H', 'neu': 'N', 'sad': 'S', 'exc': 'H'}
with open("/mnt/cephfs2/asr/database/AM/emo/iemocap/all.lst") as f:
    for i, line in enumerate(f.readlines()):
        x = line.strip().split("\t")
        if len(x) == 3:
            print(x)
            continue
        f, utt, emo, t = x
        wav_path = "/mnt/cephfs2/asr/database/AM/emo/iemocap/16k/%s.wav" % utt
        if os.path.exists(wav_path) and classes.get(emo):
            f2.write("%s %s\n"% (wav_path, classes.get(emo)))
            f3.write("%s\n"%utt)
f2.close()
f3.close()

