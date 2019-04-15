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
    path = "/Users/lonica/Documents/dev/workspace/",
    build_file = "mykaldi.BUILD",
)

new_local_repository(
    name = "torch",
    path = "/Users/lonica/Documents/dev/workspace/",
    build_file = "torch.BUILD",
)