TORCH_LIB_MAP = [
  "libcaffe2",
  "libc10",
  "libtorch",
]

MKLML_LIB_MAP = [
    "libmklml_intel",
    "libiomp5",
]

[cc_import(
    name = lib,
    # static_library = "intel/mkl/lib/lib%s.a" % lib,
    shared_library = "mklml_lnx_2019.0.3.20190220/lib/%s.so" % lib,
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
    shared_library = "libtorch/lib/%s.so" % lib,
    visibility = ["//visibility:public"],
 ) for lib in TORCH_LIB_MAP]
#cc_import(
#    name = "libtorch-shared",
#    hdrs = glob([ "libtorch/include/**/*.h"]),
#    shared_library = "libtorch/lib/libtorch.so.1",
#    visibility = ["//visibility:public"],
# )

cc_library(
  name = "libgomp",
  srcs = glob(["libtorch/lib/libgomp-8bba0e50.so.1"]),
  #hdrs = glob(["libtorch/include/**/*.h", "libtorch/include/c10/util/Exception.h"]),
  #deps = [":%s" % lib for lib in TORCH_LIB_MAP] + [ ":%s" % lib for lib in MKLML_LIB_MAP ],
  visibility = ["//visibility:public"],
)
#cc_library(
#  name = "libc10",
#  srcs = glob(["libtorch/lib/libc10.so"]),
#  hdrs = glob(["libtorch/include/c10/**/*.h"]),
#  deps = [":libgomp"],
#  visibility = ["//visibility:public"],
#)
#cc_library(
#  name = "libcaffe2",
#  srcs = glob(["libtorch/lib/libcaffe2.so"]),
#  hdrs = glob(["libtorch/include/caffe2/**/*.h"]),
#  deps = [":libgomp", ":libc10"],
#  visibility = ["//visibility:public"],
#)
cc_library(
  name = "torch",
  #srcs = glob(["libtorch/lib/libtorch.so"]),
  hdrs = glob(["libtorch/include/**/*.h"]),
  deps = [":libgomp", ":libc10", ":libcaffe2", ":libtorch"],
  visibility = ["//visibility:public"],
)
