#!/usr/bin/bash
D=$HOME/claude-lab/det; cd $D; exec > logs/setup23.log 2>&1
echo "start $(date)"; df -h $HOME | tail -1
python3 -m venv $D/venv23 && . $D/venv23/bin/activate
python -m pip install -U pip wheel 2>&1 | tail -1
python -m pip install vllm==0.23.0 2>&1 | tail -2
python -m pip install lmcache==0.5.7.post1 2>&1 | tail -2
python - <<PY
import torch, vllm, importlib.metadata as m
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("vllm", vllm.__version__, "lmcache", m.version("lmcache"))
import lmcache.cuda_ops
print("lmcache cuda_ops import OK")
PY
echo "done $(date)"
