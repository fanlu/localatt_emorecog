workspace(name = "jdemo")

# load other repositories
# load("//:bazel/mykaldi_only.bzl", "mykaldi")
# mykaldi()

new_local_repository(
    name = "mkl",
    path = "/opt/",
    build_file = "mkl.BUILD",
)

new_local_repository(
    name = "kaldi",
    path = "/Users/fanlu/workspace/",
    build_file = "mykaldi.BUILD",
)

new_local_repository(
    name = "torch",
    path = "/Users/fanlu/workspace/",
    build_file = "torch.BUILD",
)