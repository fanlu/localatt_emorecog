package(default_visibility = ["//visibility:public"])

# load("//:bazel/mykaldi_only.bzl", "kaldi_cc_binary")

# kaldi_cc_binary(
#     name="emo",
#     srcs = ["emo.cc"],
# )
# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary
cc_binary(
    name = "emo2",
    srcs = ["emo2.cc"],
    copts = [
        # "-Wno-sign-compare -Wno-unused-local-typedefs",
        # "-DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_CLAPACK ",
        "-g",
        "-Iexternal/kaldi/kaldi/src",
        "-Iexternal/kaldi/kaldi/tools/openfst/include",
        "-Iexternal/torch/libtorch/include",
        "-Iexternal/torch/mklml_mac_2019.0.3.20190220/include",
    ],
    linkopts = [
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    deps = ["@kaldi//:mykaldi", "@kaldi//:openfst", "@mkl//:mkl", "@torch//:mklml", "@torch//:torch"],
)

cc_binary(
    name = "emo",
    srcs = ["emo.cc"],
    copts = [
        # "-Wno-sign-compare -Wno-unused-local-typedefs",
        # "-DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_CLAPACK ",
        "-g",
        "-Iexternal/kaldi/kaldi/src",
        "-Iexternal/kaldi/kaldi/tools/openfst/include",
    ],
    linkopts = [
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    deps = ["@kaldi//:mykaldi", "@kaldi//:openfst", "@mkl//:mkl"],
)

cc_binary(
    name = "inference",
    srcs = ["inference.cpp"],
    copts = [
        # "-Wno-sign-compare -Wno-unused-local-typedefs",
        # "-DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_CLAPACK ",
        "-g",
        "-Iexternal/torch/libtorch/include",
        "-Iexternal/torch/mklml_mac_2019.0.3.20190220/include",
    ],
    linkopts = [
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    deps = ["@mkl//:mkl", "@torch//:mklml", "@torch//:torch"],
)