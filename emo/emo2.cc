#include <torch/script.h>  // One-stop header.

#include <iostream>
#include <memory>
#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"

int main(int argc, const char* argv[]) {
  using namespace kaldi;
  const char* usage =
      "Calculate emotion by pytorch model.\n"
      "Usage:  emo2 [options...] <path-to-exported-script-module> "
      "<wav-rspecifier>\n";
  ParseOptions po(usage);

  MfccOptions mfcc_opts;
  mfcc_opts.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 2) {
    po.PrintUsage();
    exit(1);
  }

  Mfcc mfcc(mfcc_opts);

  std::cout << "script module:" << po.GetArg(1) << std::endl;
  std::string wav_rspecifier = po.GetArg(2);
  std::cout << "wav_rspecifier:" << wav_rspecifier << std::endl;
  SequentialTableReader<WaveHolder> reader(wav_rspecifier);

  bool subtract_mean = true;
  int32 channel = -1;
  BaseFloat vtln_warp_local = 1.0;

  for (; !reader.Done(); reader.Next()) {
    std::string utt = reader.Key();
    std::cout << "utt:" << utt << std::endl;
    const WaveData& wave_data = reader.Value();
    int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
    {  // This block works out the channel (0=left, 1=right...)
      KALDI_ASSERT(num_chan > 0);  // should have been caught in
      // reading code if no channels.
      if (channel == -1) {
        this_chan = 0;
        if (num_chan != 1)
          KALDI_WARN << "Channel not specified but you have data with "
                     << num_chan << " channels; defaulting to zero";
      } else {
        if (this_chan >= num_chan) {
          KALDI_WARN << "File with id " << utt << " has " << num_chan
                     << " channels but you specified channel " << channel
                     << ", producing no output.";
          continue;
        }
      }
    }
    SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
    Matrix<BaseFloat> features;
    try {
      mfcc.ComputeFeatures(waveform, wave_data.SampFreq(), vtln_warp_local,
                           &features);
    } catch (...) {
      KALDI_WARN << "Failed to compute features for utterance " << utt;
      continue;
    }
    if (subtract_mean) {
      Vector<BaseFloat> mean(features.NumCols());
      mean.AddRowSumMat(1.0, features);
      mean.Scale(1.0 / features.NumRows());
      for (int32 i = 0; i < features.NumRows(); i++)
        features.Row(i).AddVec(-1.0, mean);
    }
    std::cout << features;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module =
      torch::jit::load(po.GetArg(1));

  assert(module != nullptr);
  std::cout << "ok\n";

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 120, 13}));
  std::vector<int> ls = {120};
  inputs.push_back(torch::tensor(ls));

  // Execute the model and turn its output into a tensor.
  // at::Tensor output = module->forward(inputs, lens).toTensor();
  at::Tensor output = module->forward(inputs).toTensor();

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
