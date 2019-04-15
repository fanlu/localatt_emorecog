utils/data/get_utt2dur.sh --nj 40 --cmd "run.pl" data/train
paste train/wav.scp train/utt2dur | awk '{if ($4 <= 16) print $1, $2}' > train_16/wav.scp
paste train/spk2gender train/utt2dur | awk '{if ($4 <= 16) print $1, $2}' > train_16/spk2gender
paste data/train/spk2utt data/train/utt2dur | awk '{if ($4 <= 16) print $1, $2}' > data/train_16/spk2utt
paste data/train/utt2spk data/train/utt2dur | awk '{if ($4 <= 16) print $1, $2}' > data/train_16/utt2spk

paste data/test/wav.scp data/test/utt2dur | awk '{if ($4 <= 16) print $1, $2}' > data/test_16/wav.scp
paste data/test/spk2utt data/test/utt2dur | awk '{if ($4 <= 16) print $1, $2}' > data/test_16/spk2utt
paste data/test/utt2spk data/test/utt2dur | awk '{if ($4 <= 16) print $1, $2}' > data/test_16/utt2spk
paste data/test/spk2gender data/test/utt2dur | awk '{if ($4 <= 16) print $1, $2}' > data/test_16/spk2gender

steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "run.pl" data/train_16 exp/make_mfcc mfcc/
steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "run.pl" data/test_16 exp/make_mfcc mfcc/