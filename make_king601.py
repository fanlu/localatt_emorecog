import os
from tqdm import tqdm
from pydub import AudioSegment
f2 = open("king601_8k/wav_cat_train.list", "w")
f3 = open("king601_8k/wav_cat_dev.list", "w")
f4 = open("king601_8k/wav_cat_test.list", "w")
#classes = {'ang': 'A', 'hap': 'H', 'neu': 'N', 'sad': 'S', 'exc': 'H'}
#classes = {'happy': 'H', 'sad': 'S', 'angry': 'A', 'surprise': 'H', '<CRY/>': 'S', '<LAUGH/>': 'H'}
classes = {'happy': 0, 'sad': 1, 'angry': 2, 'surprise': 3, '<CRY/>': 1, '<LAUGH/>': 0}
with open("/mnt/cephfs2/asr/database/AM/emo/King-ASR-601/TABLE/CONTENT.TXT") as f:
#with open("/mnt/cephfs2/asr/database/AM/emo/iemocap/all.lst") as f:
    for i, line in enumerate(tqdm(f.readlines())):
        #import pdb;pdb.set_trace()
        if i == 0:
            continue
        SCD, SES, UID, TRS, EXN = line.strip().split("\t")
        wav_path = os.path.join("/mnt/cephfs2/asr/database/AM/emo/King-ASR-601/DATA/CHANNEL0/WAVE/SPEAKER%s/SESSION%s/%s.WAV" % (SCD, SES, UID))
        #b = AudioSegment.from_raw(wav_path, sample_width=2, frame_rate=16000, channels=1)
        wav_path2 = wav_path.replace("WAVE", "WAVE_8k")
        if not os.path.exists(os.path.dirname(wav_path2)):
            os.makedirs(os.path.dirname(wav_path2))
        #b.set_frame_rate(8000).export(wav_path2, format="wav")
        emo = classes.get(EXN.split(";")[0])
        #x = line.strip().split("\t")
        #if len(x) == 3:
        #    print(x)
        #    continue
        #f, utt, emo, t = x
        #wav_path = "/mnt/cephfs2/asr/database/AM/emo/iemocap/16k/%s.wav" % utt
        if os.path.exists(wav_path2) and EXN.split(";")[0] in classes:
            if SCD <= '0230':
                f2.write("%s %s\n"% (wav_path2, emo))
            elif SCD <= '0240':
                f3.write("%s %s\n"% (wav_path2, emo))
            elif SCD <= '0250':
                f4.write("%s %s\n"% (wav_path2, emo))
f2.close()
f3.close()
f4.close()

