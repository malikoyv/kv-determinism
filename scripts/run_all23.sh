#!/usr/bin/bash
# All arms on vLLM 0.23 + LMCache 0.5.7 (source-built ops). Sequential; one STATUS line per arm.
D=$HOME/claude-lab/det; cd $D; export VENV=venv23 VLLM_USE_FLASHINFER_SAMPLER=0
. venv23/bin/activate; rm -f STATUS_all
plain() { python gen.py $1 res/v23_$2.json > logs/v23_$2.log 2>&1; echo "STATUS $2 rc=$? $(date)" >> STATUS_all; }
mp()    { /usr/bin/bash run_mp.sh $1 v23_$2; tail -1 STATUS_mp | sed "s/^/ALL /" >> STATUS_all; }
plain r r1; plain r r2; plain g g1; plain g g2
mp d d2; mp s s1; mp s s2
echo "STATUS ALL DONE $(date)" >> STATUS_all
