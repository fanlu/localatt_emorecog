workspace(name = "jdemo")

# load other repositories
# load("//:bazel/mykaldi_only.bzl", "mykaldi")
# mykaldi()

new_local_repository(
    name = "mkl",
    path = "/opt/",
    build_file = "mkl_linux.BUILD",
)

new_local_repository(
    name = "kaldi",
    path = "/mnt/cephfs2/asr/users/fanlu/",
    build_file = "mykaldi.BUILD",
)

new_local_repository(
    name = "torch",
    path = "/mnt/cephfs2/asr/users/fanlu/",
    build_file = "torch_linux.BUILD",
)
