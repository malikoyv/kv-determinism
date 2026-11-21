# Does a KV-cache hit change the answer? GPU cache vs LMCache DRAM vs LMCache SSD

Greedy decoding (temperature 0), batch size 1, same 40 prompts, one RTX 5070 Ti (16 GB).
Question: is the output after a cache hit the same whether the KV came from the GPU prefix cache,
DRAM, or SSD, and how does it compare with recomputing the prefix?

## Setup (what actually ran)

| Item | Value |
|---|---|
| Model | Qwen/Qwen2.5-3B-Instruct, bf16 |
| Engine | vLLM 0.23.0, torch 2.11.0+cu130, attention backend FLASH_ATTN, `VLLM_USE_FLASHINFER_SAMPLER=0` |
| KV tier | LMCache 0.5.7.post1 **built from source** (the published wheel's `cuda_ops` fails to load on torch 2.11), multiprocess server |
| DRAM arm | `lmcache server --l1-size-gb 8` |
| SSD arm | `--l1-size-gb 4 --l2-store-policy skip_l1 --l2-adapter {"type":"fs_native",...,"use_odirect":true}` |
| Prompts | 5 contexts (3K-token stdlib source files) x 8 questions. Question 1 per context is a cache miss, questions 2-8 are hits |

## Run

```
bash setup23.sh                 # venv with vllm==0.23.0 + lmcache wheel
bash build_lmc.sh               # rebuild lmcache ops against torch 2.11 (needs nvcc 13.0.x from pip)
bash run_all23.sh               # R x2, G x2, D, S x2   (run_mp.sh starts/stops the LMCache server)
bash run_chunk.sh               # D at LMCache chunk size 128, 512 (64 was run by hand with CHUNK=64)
python boundary.py              # the headline comparison
```

`gen.py ARM OUT.json` arms: `r` recompute, `g` vLLM GPU prefix cache, `d` LMCache DRAM, `s` LMCache SSD.
A d/s run is rejected (`*.INVALID`) if no hit prompt reports cached tokens, because LMCache silently
falls back to recompute when its server is unhealthy.

## Results (raw files in `results/`)

| Check | Result |
|---|---|
| Each arm vs its own repeat (R, G, D, S) | 40/40 bit-identical |
| Miss prompts, every arm vs recompute | 5/5 bit-identical |
| Hit prompts diverging from recompute | GPU cache 23/35, DRAM 22/35, SSD 22/35 (chunk 256) |
| DRAM vs SSD, hit prompts | 35/35 bit-identical |
| LMCache chunk size 64 / 128 / 256 / 512 | outputs change with chunk size (e.g. 64 vs 512: 32/35 differ) |
| Same reused length, any pair of 6 runs | **173/173** bit-identical |
| Different reused length, any pair of 6 runs | **0/352** bit-identical |

SSD arm evidence: the LMCache server logged 35 prefetches per run, all `0 L1, N L2`.

## Limits (measured vs not)

- One model (3B), one GPU, 5 contexts: the 35 hit prompts are not independent samples.
- Relative to a no-cache recompute, 7/35 hit prompts were bit-identical at some reused lengths
  (all in one context per arm), so "different length gives different bits" holds between hits,
  not against the recompute run.
- Not done: hashing the loaded KV tensors (D = S shows identical behaviour, not identical bytes);
  any accuracy or quality measurement; other models, bigger contexts, fp8 KV; chunk-size runs were
  executed once each (repeatability was verified at chunk 256 and for all other arms).
- The mechanism (the cache boundary changes the attention kernel's tile and summation order) is a
  hypothesis consistent with the data, not a measured cause.
