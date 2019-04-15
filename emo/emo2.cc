#include <torch/script.h>  // One-stop header.

#include <iostream>
#include <memory>
#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr
        << "usage: example-app <path-to-exported-script-module> wav.scp\n";
    return -1;
  }

  using namespace kaldi;
  const char *usage =
      "Create MFCC feature files.\n"
      "Usage:  compute-mfcc-feats [options...] <wav-rspecifier> "
      "<feats-wspecifier>\n";
  ParseOptions po(usage);

  // Deserialize the ScriptModule from a file using torch::jit::load().
  // std::shared_ptr<torch::jit::script::Module> module =
  //     torch::jit::load(argv[1]);

  // assert(module != nullptr);
  std::cout << "ok\n";

  // std::string wav_rspecifier = argv[2];
  // SequentialTableReader<WaveHolder> reader(wav_rspecifier);
  // bool subtract_mean = true;
  // int32 channel = -1;
  // BaseFloat vtln_warp_local = 1.0;;
  // MfccOptions mfcc_opts;
  // mfcc_opts.Register(&po);
  // Mfcc mfcc(mfcc_opts);
  // for (; !reader.Done(); reader.Next()) {
  //   std::string utt = reader.Key();
  //   const WaveData& wave_data = reader.Value();
  //   int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
  //   SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
  //   Matrix<BaseFloat> features;
  //   try {
  //     mfcc.ComputeFeatures(waveform, wave_data.SampFreq(), vtln_warp_local,
  //                          &features);
  //   } catch (...) {
  //     KALDI_WARN << "Failed to compute features for utterance " << utt;
  //     continue;
  //   }
  //   if (subtract_mean) {
  //     Vector<BaseFloat> mean(features.NumCols());
  //     mean.AddRowSumMat(1.0, features);
  //     mean.Scale(1.0 / features.NumRows());
  //     for (int32 i = 0; i < features.NumRows(); i++)
  //       features.Row(i).AddVec(-1.0, mean);
  //   }
  //   std::cout << features;
  // }

  // std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(torch::ones({1, 120, 13}));
  // std::vector<int> ls = {120};
  // inputs.push_back(torch::tensor(ls));

  // // Execute the model and turn its output into a tensor.
  // // at::Tensor output = module->forward(inputs, lens).toTensor();
  // at::Tensor output = module->forward(inputs).toTensor();

  // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
