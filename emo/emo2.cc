#include <torch/script.h>  // One-stop header.

#include <iostream>
#include <memory>
#include <chrono>
#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"

int main(int argc, const char* argv[]) {
  using namespace kaldi;
  const char* usage =
      "Calculate emotion by pytorch model.\n"
      "Usage:  emo2 [options...] <path-to-exported-script-module> "
      "<wav-rspecifier> <flag-rspecifier>\n";
  ParseOptions po(usage);

  MfccOptions mfcc_opts;
  mfcc_opts.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 3) {
    po.PrintUsage();
    exit(1);
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  //std::shared_ptr<torch::jit::script::Module> module =
  //    torch::jit::load(po.GetArg(1));
  // module->to(torch::kHalf);
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(po.GetArg(1));
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  //assert(module != nullptr);
  std::cout << "ok\n";

  Mfcc mfcc(mfcc_opts);

  std::cout << "script module:" << po.GetArg(1) << std::endl;
  std::string wav_rspecifier = po.GetArg(2);
  std::cout << "wav_rspecifier:" << wav_rspecifier << std::endl;
  SequentialTableReader<WaveHolder> reader(wav_rspecifier);
  std::string flag_wspecifier = po.GetArg(3);
  BaseFloatWriter flag_writer(flag_wspecifier);
  std::cout << "save to file:" << flag_wspecifier << std::endl;

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

    std::vector<torch::jit::IValue> inputs;
    if (subtract_mean) {
      Vector<BaseFloat> mean(features.NumCols());
      mean.AddRowSumMat(1.0, features);
      mean.Scale(1.0 / features.NumRows());

      for (int32 i = 0; i < features.NumRows(); i++) {
        features.Row(i).AddVec(-1.0, mean);
      }
    }

    // at::ArrayRef sizes({1, 120, 13});
    // std::cout << features;
    // inputs.push_back(torch::ones({1, 120, 13}));
    // std::cout << "========" << features.Stride() << std::endl;
    torch::Tensor input = torch::from_blob(
        features.Data(), {1, features.NumRows(), features.NumCols()},
        {features.NumRows() * features.Stride(), features.Stride(), 1});
    input.to(torch::kHalf);
    inputs.push_back(input);
    // std::cout << input.sizes() << std::endl;

    std::vector<int> ls = {features.NumRows()};
    inputs.push_back(torch::tensor(ls).to(torch::kHalf));

    // Execute the model and turn its output into a tensor.
    // at::Tensor output = module->forward(inputs, lens).toTensor();
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor output = module.forward(inputs).toTensor();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "max index:" << output.argmax(1).item<int64_t>() << std::endl;
    std::cout << "input shape:" << input.sizes() << ", C++ Operation Time(s) " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
    auto max_result = output.max(1, true);
    auto max_index = std::get<1>(max_result).item<float>();
    // std::cout << "max index2:" << max_index << std::endl;
    // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    flag_writer.Write(utt, max_index);
  }
}
