export KALDI_ROOT=/asr_storage/fanlu/kaldi2
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

source /asr_storage/fanlu/miniconda3/bin/activate && conda deactivate && conda activate py3
export PATH=/asr_storage/fanlu/espnet/utils:$PATH
export PYTHONPATH=$KALDI_ROOT/src/pybind:$PYTHONPATH