// featbin/compute-mfcc-feats.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "ivector/voice-activity-detection.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "gmm/mle-full-gmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[])
{
    using namespace kaldi;
    const char *usage =
        "Create MFCC feature files.\n"
        "Usage:  compute-mfcc-feats [options...] <wav-rspecifier> <feats-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    MfccOptions mfcc_opts;
    bool subtract_mean = false;
    BaseFloat vtln_warp = 1.0;
    int32 channel = -1;
    std::string utt2spk_rspecifier;
    std::string vtln_map_rspecifier;

    // Register the MFCC option struct
    mfcc_opts.Register(&po);

    // Register the options
    // po.Register("output-format", &output_format, "Format of the output "
    //             "files [kaldi, htk]");
    po.Register("subtract-mean", &subtract_mean, "Subtract mean of each "
                                                 "feature file [CMS]; not recommended to do it this way. ");
    po.Register("vtln-warp", &vtln_warp, "Vtln warp factor (only applicable "
                                         "if vtln-map not specified)");
    po.Register("vtln-map", &vtln_map_rspecifier, "Map from utterance or "
                                                  "speaker-id to vtln warp factor (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier, "Utterance to speaker-id map "
                                                "rspecifier (if doing VTLN and you have warps per speaker)");
    // po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, "
    //             "0 -> left, 1 -> right)");
    // po.Register("min-duration", &min_duration, "Minimum duration of segments "
    //             "to process (in seconds).");

    bool omit_unvoiced_utts = false;
    po.Register("omit-unvoiced-utts", &omit_unvoiced_utts,
                "If true, do not write out voicing information for "
                "utterances that were judged 100% unvoiced.");
    VadEnergyOptions vad_opts;
    vad_opts.Register(&po);

    DeltaFeaturesOptions delta_opts;
    int32 truncate = 0;
    po.Register("truncate", &truncate, "If nonzero, first truncate features to this dimension.");
    delta_opts.Register(&po);

    SlidingWindowCmnOptions cmvn_opts;
    cmvn_opts.Register(&po);

    // gmm-gselect
    std::string gselect_rspecifier;
    std::string likelihood_wspecifier;
    po.Register("write-likes", &likelihood_wspecifier, "rspecifier for likelihoods per "
                "utterance");
    po.Register("gselect", &gselect_rspecifier, "rspecifier for gselect objects "
                "to limit the search to");

    po.Read(argc, argv);

    if (po.NumArgs() != 5)
    {
        po.PrintUsage();
        exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1);
    std::string model_filename = po.GetArg(2);
    std::string f_model_filename = po.GetArg(3);
    std::string male_model_filename = po.GetArg(4);
    std::string female_model_filename = po.GetArg(5);

    Mfcc mfcc(mfcc_opts);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    RandomAccessBaseFloatReaderMapped vtln_map_reader(vtln_map_rspecifier,
                                                      utt2spk_rspecifier);

    for (; !reader.Done(); reader.Next())
    {
        std::string utt = reader.Key();
        const WaveData &wave_data = reader.Value();
        int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
        {                               // This block works out the channel (0=left, 1=right...)
            KALDI_ASSERT(num_chan > 0); // should have been caught in
            // reading code if no channels.
            if (channel == -1)
            {
                this_chan = 0;
                if (num_chan != 1)
                    KALDI_WARN << "Channel not specified but you have data with "
                               << num_chan << " channels; defaulting to zero";
            }
            else
            {
                if (this_chan >= num_chan)
                {
                    KALDI_WARN << "File with id " << utt << " has "
                               << num_chan << " channels but you specified channel "
                               << channel << ", producing no output.";
                    continue;
                }
            }
        }
        BaseFloat vtln_warp_local; // Work out VTLN warp factor.
        if (vtln_map_rspecifier != "")
        {
            if (!vtln_map_reader.HasKey(utt))
            {
                KALDI_WARN << "No vtln-map entry for utterance-id (or speaker-id) "
                           << utt;
                continue;
            }
            vtln_warp_local = vtln_map_reader.Value(utt);
        }
        else
        {
            vtln_warp_local = vtln_warp;
        }

        SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
        Matrix<BaseFloat> features;
        try
        {
            mfcc.ComputeFeatures(waveform, wave_data.SampFreq(), vtln_warp_local, &features);
        }
        catch (...)
        {
            KALDI_WARN << "Failed to compute features for utterance "
                       << utt;
            continue;
        }
        if (subtract_mean)
        {
            Vector<BaseFloat> mean(features.NumCols());
            mean.AddRowSumMat(1.0, features);
            mean.Scale(1.0 / features.NumRows());
            for (int32 i = 0; i < features.NumRows(); i++)
                features.Row(i).AddVec(-1.0, mean);
        }
        std::cout << features;

        // compute vad
        Vector<BaseFloat> vad_result(features.NumRows());

        ComputeVadEnergy(vad_opts, features, &vad_result);
        int32 num_unvoiced = 0;
        double tot_length = 0.0, tot_decision = 0.0;
        double sum = vad_result.Sum();
        if (sum == 0.0) {
            KALDI_WARN << "No frames were judged voiced for utterance " << utt;
            num_unvoiced++;
        }
        tot_decision += vad_result.Sum();
        tot_length += vad_result.Dim();
        std::cout << " " << tot_decision << " " << tot_length;

        // compute delta
        Matrix<BaseFloat> new_feats;
        if (truncate != 0) {
            if (truncate > features.NumCols())
            KALDI_ERR << "Cannot truncate features as dimension " << features.NumCols()
                        << " is smaller than truncation dimension.";
            SubMatrix<BaseFloat> feats_sub(features, 0, features.NumRows(), 0, truncate);
            ComputeDeltas(delta_opts, feats_sub, &new_feats);
        } else {
            ComputeDeltas(delta_opts, features, &new_feats);
        }
        std::cout << new_feats;

        // apply-cmvn-sliding
        Matrix<BaseFloat> cmvn_feat(new_feats.NumRows(),
                                  new_feats.NumCols(), kUndefined);

        SlidingWindowCmn(cmvn_opts, new_feats, &cmvn_feat);
        std::cout << cmvn_feat;

        // select-voiced-frames 
        if (cmvn_feat.NumRows() != vad_result.Dim()) {
            KALDI_WARN << "Mismatch in number for frames " << cmvn_feat.NumRows()
                    << " for features and VAD " << vad_result.Dim();
            continue;
        }
        if (vad_result.Sum() == 0.0) {
            KALDI_WARN << "No features were judged as voiced for utterance ";
            continue;
        }
        int32 dim = 0;
        for (int32 i = 0; i < vad_result.Dim(); i++){
            if (vad_result(i) != 0.0){
                dim++;
            }
        }
            
        Matrix<BaseFloat> voiced_feat(dim, cmvn_feat.NumCols());
        int32 index = 0;
        for (int32 i = 0; i < cmvn_feat.NumRows(); i++) {
            if (vad_result(i) != 0.0) {
                KALDI_ASSERT(vad_result(i) == 1.0); // should be zero or one.
                voiced_feat.Row(index).CopyFromVec(cmvn_feat.Row(i));
                index++;
            }
        }
        std::cout << voiced_feat;

        // --n=20 --n=3 in gender_id.sh
        int32 num_gselect1 = 20, num_gselect2 = 3;

        // gmm-gselect
        DiagGmm gmm;
        ReadKaldiObject(model_filename, &gmm);
        KALDI_ASSERT(num_gselect1 > 0);
        int32 num_gauss = gmm.NumGauss();
        if (num_gselect1 > num_gauss) {
            KALDI_WARN << "You asked for " << num_gselect1 << " Gaussians but GMM "
                        << "only has " << num_gauss << ", returning this many. "
                        << "Note: this means the Gaussian selection is pointless.";
            num_gselect1 = num_gauss;
        }
        
        int32 tot_t_this_file = 0; double tot_like_this_file = 0;
        std::vector<std::vector<int32> > gselect(voiced_feat.NumRows());
        tot_t_this_file += voiced_feat.NumRows();
        tot_like_this_file = gmm.GaussianSelection(voiced_feat, num_gselect1, &gselect);

        std::cout << "";

        // fgmm-gselect
        FullGmm fgmm;
        ReadKaldiObject(f_model_filename, &fgmm);
        KALDI_ASSERT(num_gselect2 > 0);
        int32 num_gauss_f = fgmm.NumGauss();
        KALDI_ASSERT(num_gauss_f);
        if (num_gselect2 > num_gauss_f) {
            KALDI_WARN << "You asked for " << num_gselect2 << " Gaussians but GMM "
                        << "only has " << num_gauss_f << ", returning this many. "
                        << "Note: this means the Gaussian selection is pointless.";
            num_gselect2 = num_gauss_f;
        }
        std::vector<std::vector<int32> > gselect_f(voiced_feat.NumRows());
        tot_t_this_file += voiced_feat.NumRows();
        // Limit Gaussians to preselected group...
        // gselect
        // const vector<vector<int32> > &preselect = gselect_reader.Value(utt);
        if (gselect.size() != static_cast<size_t>(voiced_feat.NumRows())) {
            KALDI_WARN << "Input gselect for utterance " << utt << " has wrong size "
                        << gselect.size() << " vs. " << voiced_feat.NumRows();
            continue;
        }
        for (int32 i = 0; i < voiced_feat.NumRows(); i++)
            tot_like_this_file +=
                fgmm.GaussianSelectionPreselect(voiced_feat.Row(i), gselect[i],
                                                num_gselect2, &(gselect_f[i]));

        // fgmm-global-get-frame-likes
        bool average = true;
        FullGmm male_fgmm, female_fgmm;
        {
            bool binary_read;
            Input ki(male_model_filename, &binary_read), ki2(female_model_filename, &binary_read);
            male_fgmm.Read(ki.Stream(), binary_read);
            female_fgmm.Read(ki2.Stream(), binary_read);
        }
        int32 file_frames = voiced_feat.NumRows();
        Vector<BaseFloat> likes(file_frames), likes_f(file_frames);
        for (int32 i = 0; i < file_frames; i++) {
            SubVector<BaseFloat> data(voiced_feat, i);
            const std::vector<int32> &this_gselect = gselect_f[i];
            int32 gselect_size = this_gselect.size();
            KALDI_ASSERT(gselect_size > 0);
            Vector<BaseFloat> loglikes, loglikes_f;
            male_fgmm.LogLikelihoodsPreselect(data, this_gselect, &loglikes);
            female_fgmm.LogLikelihoodsPreselect(data, this_gselect, &loglikes_f);
            likes(i) = loglikes.LogSumExp();
            likes_f(i) = loglikes_f.LogSumExp();
        }
        if (average){
            float male = likes.Sum() / file_frames;
            float female = likes_f.Sum() / file_frames;
            std::cout << "male:" << male << " female:" << female;
            float pmale = 0.5;
            float lratio = log(pmale/(1-pmale))+male-female;
            float result = 1/(1+exp(-lratio));
            std::cout << "result:" << result << std::endl;
            if (result > 0.5) {
                std::cout << "male";
            } else {
                std::cout << "female";
            }
        }
        
            
    }
    return 0;
}