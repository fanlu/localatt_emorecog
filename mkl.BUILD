# Starting from root of all the third_party packages.

# ##########################################################################
# mkl libraries
MKL_LIB_MAP = [
    "mkl_intel_lp64",
    "mkl_core",
    "mkl_sequential",
]

# TODO(xuedong): Figure out how to link static with bazel.
# somehow the mkl static linke does not work with bazel (ld.gold)
# even the dynamic link is very slow
[cc_import(
    name = lib,
    static_library = "intel/mkl/lib/lib%s.a" % lib,
    # shared_library = "intel/mkl/lib/lib%s.so" % lib,
    visibility = ["//visibility:public"],
 ) for lib in MKL_LIB_MAP]

cc_library(
  name = "mkl",
  deps = [ ":%s" % lib for lib in MKL_LIB_MAP ],
  visibility = ["//visibility:public"],
)
