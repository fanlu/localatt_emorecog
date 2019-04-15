OPENFST_LIB_MAP = [
    "libfst",
    "libfstfar",
    "libfstfarscript",
    "libfstngram",
    "libfstscript",
]

[cc_import(
    name = "openfst-" + lib,
    hdrs =glob([
        "kaldi/tools/openfst/include/**/*.h",
    ]),
    static_library = "kaldi/tools/openfst/lib/%s.a" % lib,
    # shared_library = ...
    visibility = ["//visibility:public"],
 ) for lib in OPENFST_LIB_MAP]

cc_library(
  name = "openfst",
  hdrs = glob(
    include = [
        "kaldi/tools/openfst/include/**/*.h",
    ],
  ),
  deps = [ ":openfst-%s" % lib for lib in OPENFST_LIB_MAP ],
  visibility = ["//visibility:public"],
)

[cc_import(
    name = "shared-openfst-" + lib,
    hdrs =glob([
        "kaldi/tools/openfst/include/**/*.h",
    ]),
    shared_library = "kaldi/tools/openfst/lib/%s.so" % lib,
    # shared_library = ...
    visibility = ["//visibility:public"],
 ) for lib in OPENFST_LIB_MAP]

cc_library(
  name = "openfst-shared",
  hdrs = glob(
    include = [
        "kaldi/tools/openfst/include/**/*.h",
    ],
  ),
  deps = [ ":shared-openfst-%s" % lib for lib in OPENFST_LIB_MAP ],
  visibility = ["//visibility:public"],
)

KALDI_LIB_MAP = [
    "online2",
    "ivector",
    "nnet3",
    "chain",
    "nnet2",
    "cudamatrix",
    "decoder",
    "lat",
    "fstext",
    "hmm",
    "feat",
    "transform",
    "gmm",
    "tree",
    "util",
    "matrix",
    "base",
]

[cc_import(
    name = "kaldi-" + lib,
    hdrs = glob([ "kaldi/src/%s/*.h" % lib ]),
    static_library = "kaldi/src/%s/kaldi-%s.a" % (lib, lib),
    # shared_library = ...
    visibility = ["//visibility:public"],
 ) for lib in KALDI_LIB_MAP]

cc_library(
  name = "mykaldi",
  hdrs = glob(
    include = [
        "kaldi/src/*/*.h",
    ],
  ),
  # copts=["-Iexternal/mykaldi/include"],
#   copts = [
#       " -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_CLAPACK -framework Accelerate -lm -lpthread -ldl ",
#         "-Wno-sign-compare -Wno-unused-local-typedefs",
#         "-Iexternal/kaldi/mykaldi/src",
#         "-Iexternal/kaldi/mykaldi/tools/openfst/include",
#     ],
  deps = 
  # kaldi_math_deps() + 
    [ ":openfst" ] +
    [ ":kaldi-%s" % lib for lib in KALDI_LIB_MAP ],
  visibility = ["//visibility:public"],
)

