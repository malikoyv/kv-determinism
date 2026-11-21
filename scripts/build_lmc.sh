#!/usr/bin/bash
# Build LMCache 0.5.7.post1 from source against torch 2.11/cu130 in venv23 (nvcc 13.0.88 from pip).
D=$HOME/claude-lab/det; cd $D; exec > logs/build_lmc.log 2>&1
. venv23/bin/activate
echo "start $(date)"; free -g | sed -n 2p
python -m pip install "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.88" "nvidia-cuda-crt==13.0.88" 2>&1 | tail -2
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export CUDA_HOME=$SP/nvidia/cu13
[ -e $CUDA_HOME/lib64 ] || ln -s lib $CUDA_HOME/lib64
ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
$CUDA_HOME/bin/nvcc --version | tail -2
ls $CUDA_HOME/include | grep -E "^cuda_runtime.h$|^cuda.h$" | tr "\n" " "; echo
export PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=6
python -m pip install setuptools_scm ninja packaging wheel setuptools "grpcio-tools<1.79" "protobuf>=6.31.1,<7" cmake pybind11 2>&1 | tail -2
python -m pip install --no-binary lmcache --no-build-isolation --no-deps --force-reinstall -v lmcache==0.5.7.post1 > logs/build_full.log 2>&1; echo pip-rc=$?; grep -nE 'error|Error|FAILED' logs/build_full.log | head -15 | cut -c1-260
python - <<PY
import importlib; m = importlib.import_module("lmcache.cuda_ops"); print("cuda_ops import OK", m.__file__)
PY
echo "done $(date)"
