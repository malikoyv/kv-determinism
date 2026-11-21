#!/usr/bin/bash
D=$HOME/claude-lab/det; cd $D; export VENV=venv23 VLLM_USE_FLASHINFER_SAMPLER=0; rm -f STATUS_chunk
for c in 128 512; do CHUNK=$c /usr/bin/bash run_mp.sh d v23_dc$c; echo "CHUNK $c $(tail -1 STATUS_mp)" >> STATUS_chunk; done
echo "CHUNK ALL DONE $(date)" >> STATUS_chunk
