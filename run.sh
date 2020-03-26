#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# It is closely based on "X-vectors: Robust DNN Embeddings for Speaker
# Recognition" by Snyder et al.  In the future, we will add score-normalization
# and a more effective form of PLDA domain adaptation.
#
# Pretrained models are available for this recipe.  See
# http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# SRE16 trials
sre16_trials=data/sre16_eval_test/trials
sre16_trials_tgl=data/sre16_eval_test/trials_tgl
sre16_trials_yue=data/sre16_eval_test/trials_yue
nnet_dir=exp/xvector_nnet_1a

stage=0
train_stage=-1
sample=16k
num_epochs=6
lr=1e-3

. utils/parse_options.sh
if [ $stage -le 0 ]; then
  # # Path to some, but not all of the training corpora
  # data_root=/export/corpora/LDC

  # # Prepare telephone and microphone speech from Mixer6.
  # local/make_mx6.sh $data_root/LDC2013S03 data/

  # # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # # NOTE: This corpus is now available through the LDC as LDC2017S06.
  # local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/

  # # Prepare SRE08 test and enroll. Includes some microphone speech.
  # local/make_sre08.pl $data_root/LDC2011S08 $data_root/LDC2011S05 data/

  # # This prepares the older NIST SREs from 2004-2006.
  # local/make_sre.sh $data_root data/

  # # Combine all SREs prior to 2016 and Mixer6 into one dataset
  # utils/combine_data.sh data/sre \
  #   data/sre2004 data/sre2005_train \
  #   data/sre2005_test data/sre2006_train \
  #   data/sre2006_test_1 data/sre2006_test_2 \
  #   data/sre08 data/mx6 data/sre10
  # utils/validate_data_dir.sh --no-text --no-feats data/sre
  # utils/fix_data_dir.sh data/sre

  # # Prepare SWBD corpora.
  # local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
  #   data/swbd_cellular1_train
  # local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
  #   data/swbd_cellular2_train
  # local/make_swbd2_phase1.pl $data_root/LDC98S75 \
  #   data/swbd2_phase1_train
  # local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
  #   data/swbd2_phase2_train
  # local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
  #   data/swbd2_phase3_train

  # # Combine all SWB corpora into one dataset.
  # utils/combine_data.sh data/swbd \
  #   data/swbd_cellular1_train data/swbd_cellular2_train \
  #   data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # # Prepare NIST SRE 2016 evaluation data.
  # local/make_sre16_eval.pl /export/corpora5/SRE/R149_0_1 data

  # # Prepare unlabeled Cantonese and Tagalog development data. This dataset
  # # was distributed to SRE participants.
  # local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data
  python local/make_iemocap.py 16k
  for x in train test; do
    for file in wav.scp utt2spk; do
      sort data/iemocap_${sample}/$x/$file -o data/iemocap_${sample}/$x/$file
    done
    utils/utt2spk_to_spk2utt.pl data/iemocap_${sample}/$x/utt2spk > data/iemocap_${sample}/$x/spk2utt
    sort data/iemocap_${sample}/$x/spk2utt -o data/iemocap_${sample}/$x/spk2utt
    echo "data/iemocap_${sample}/$x"
    utils/data/validate_data_dir.sh --no-text --no-feats data/iemocap_${sample}/$x || exit 1;
  done

  # rm -r $data/wav.flist

  utils/data/combine_data.sh --extra-files utt2emo data/train_${sample} data/iemocap_${sample}/train/
  sort data/train_${sample}/utt2spk -o data/train_${sample}/utt2spk
  utils/utt2spk_to_spk2utt.pl data/train_${sample}/utt2spk > data/train_${sample}/spk2utt
  utils/fix_data_dir.sh data/train_${sample}
fi
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl \
      /export/b{14,15,16,17}/$USER/kaldi-data/egs/sre16/v2/xvector-$(date +'%m_%d_%H_%M')/mfccs/storage $mfccdir/storage
  fi
  utils/data/perturb_data_dir_speed_3way2.sh data/train_${sample} data/train_${sample}_sp
  # cat data/train_16k_sp/utt2spk | awk '{gsub("sp0.9-","",$2);gsub("sp1.1-","",$2);printf("%s %s\n", $1, $2)}' > data/train_16k_sp/utt2spk2
  # mv data/train_16k_sp/utt2spk data/train_16k_sp/utt2spk.bak
  # mv data/train_16k_sp/utt2spk2 data/train_16k_sp/utt2spk
  # utils/utt2spk_to_spk2utt.pl data/train_${sample}_sp/utt2spk > data/train_${sample}_sp/spk2utt
  # utils/fix_data_dir.sh data/train_${sample}_sp
  for name in train_16k train_16k_sp iemocap_16k/test; do
  # for name in train_${sample}; do
    # steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 4 --cmd "$train_cmd" \
    #   data/${name} exp/make_mfcc $mfccdir
    steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" --nj 200 data/${name}
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
  # utils/combine_data.sh --extra-files "utt2num_frames" data/train_${sample} data/swbd data/sre
  # utils/fix_data_dir.sh data/train_${sample}
fi

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train_${sample}/utt2num_frames > data/train_${sample}/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train_${sample} data/train_${sample}_reverb
  cp data/train_${sample}/vad.scp data/train_${sample}_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_${sample}_reverb data/train_${sample}_reverb.new
  rm -rf data/train_${sample}_reverb
  mv data/train_${sample}_reverb.new data/train_${sample}_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  # steps/data/make_musan.sh --sampling-rate 8000 /export/corpora/JHU/musan data
  steps/data/make_musan.sh --sampling-rate 16000 /mnt/cfs2/asr/database/SpkRecog/16k/database/musan/ data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train_${sample} data/train_${sample}_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train_${sample} data/train_${sample}_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train_${sample} data/train_${sample}_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh --extra-files utt2emo data/train_${sample}_aug data/train_${sample}_reverb data/train_${sample}_noise data/train_${sample}_music data/train_${sample}_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  # utils/subset_data_dir.sh data/train_${sample}_aug 128000 data/train_${sample}_aug_128k
  # utils/fix_data_dir.sh data/train_${sample}_aug_128k

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 200 --cmd "$train_cmd" \
  #   data/train_${sample}_aug_128k exp/make_mfcc $mfccdir
  steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" --nj 500 data/train_${sample}_aug

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_${sample}_combined data/train_${sample}_aug data/train_${sample}_sp

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  #utils/copy_data_dir.sh data/train_${sample}_combined data/sre_combined
  #utils/filter_scp.pl data/sre/spk2utt data/train_${sample}_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_combined/utt2spk
  #utils/fix_data_dir.sh data/sre_combined

fi

<<not_used
# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 4 --cmd "$train_cmd" \
    data/train_${sample}_combined data/train_${sample}_combined_no_sil exp/train_${sample}_combined_no_sil
  utils/fix_data_dir.sh data/train_${sample}_combined_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=150
  mv data/train_${sample}_combined_no_sil/utt2num_frames data/train_${sample}_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_${sample}_combined_no_sil/utt2num_frames.bak > data/train_${sample}_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_${sample}_combined_no_sil/utt2num_frames data/train_${sample}_combined_no_sil/utt2spk > data/train_${sample}_combined_no_sil/utt2spk.new
  mv data/train_${sample}_combined_no_sil/utt2spk.new data/train_${sample}_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_${sample}_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_${sample}_combined_no_sil/spk2utt > data/train_${sample}_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_${sample}_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_${sample}_combined_no_sil/spk2utt > data/train_${sample}_combined_no_sil/spk2utt.new
  mv data/train_${sample}_combined_no_sil/spk2utt.new data/train_${sample}_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_${sample}_combined_no_sil/spk2utt > data/train_${sample}_combined_no_sil/utt2spk

  utils/filter_scp.pl data/train_${sample}_combined_no_sil/utt2spk data/train_${sample}_combined_no_sil/utt2num_frames > data/train_${sample}_combined_no_sil/utt2num_frames.new
  mv data/train_${sample}_combined_no_sil/utt2num_frames.new data/train_${sample}_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_${sample}_combined_no_sil
fi
not_used
# local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage $train_stage \
#   --data data/train_${sample}_combined_no_sil --nnet-dir $nnet_dir --num_epochs $num_epochs  \
#   --egs-dir $nnet_dir/egs

if [ $stage -le 4 ]; then
    ${cuda_cmd} --gpu 1 $nnet_dir/train.log \
        python run3.py \
        --dataset data/train_16k_aug \
        --test_dataset data/iemocap_16k/test/ \
        --exp_dir $nnet_dir \
        --feat_type kaldi_mfcc \
        --feat_dim 43 \
        --lr $lr
fi

if [ $stage -le 7 ]; then
  local/extract_emo.sh --cmd "queue.pl --mem 6G" --nj 4 $nnet_dir data/iemocap_16k/test/ exp/emo_iemocap_16k_test
fi
