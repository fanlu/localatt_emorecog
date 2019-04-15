TORCH_LIB_MAP = [
  "libc10",
  "libcaffe2",
  "libtorch",
  "libtorch.1"
]

MKLML_LIB_MAP = [
    "libmklml",
    "libiomp5",
]

[cc_import(
    name = lib,
    # static_library = "intel/mkl/lib/lib%s.a" % lib,
    shared_library = "mklml_mac_2019.0.3.20190220/lib/%s.dylib" % lib,
    visibility = ["//visibility:public"],
 ) for lib in MKLML_LIB_MAP]

cc_library(
  name = "mklml",
  deps = [ ":%s" % lib for lib in MKLML_LIB_MAP ],
  visibility = ["//visibility:public"],
)

[cc_import(
    name = lib,
    hdrs = glob([ "libtorch/include/**/*.h"]),
    # static_library = "libtorch/src/%s/kaldi-%s.a" % (lib, lib),
    shared_library = "libtorch/lib/%s.dylib" % lib,
    visibility = ["//visibility:public"],
 ) for lib in TORCH_LIB_MAP]

cc_library(
  name = "torch",
  hdrs = glob(["libtorch/include/**/*.h"]),
  deps = [":%s" % lib for lib in TORCH_LIB_MAP] + [ ":%s" % lib for lib in MKLML_LIB_MAP ],
  visibility = ["//visibility:public"],
)