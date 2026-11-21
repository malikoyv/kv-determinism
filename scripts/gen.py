"""Greedy generation at batch size 1, dump token ids + logprobs. Arms: r (no cache) first.
usage: python gen.py ARM OUT.json   (ARM: r | g ; lmcache arms added later)"""
import sys, json, glob, os
arm, out = sys.argv[1], sys.argv[2]
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")
CTX_TOKENS_CHARS = 9000   # ~3K tokens of stdlib source per context
N_CTX, N_Q = 5, 8

def prompts():
    # deterministic: stdlib source files as contexts, fixed questions. Same for every arm.
    import json as _j, textwrap, difflib, argparse, re
    files = [os.path.__file__, _j.__file__, textwrap.__file__, difflib.__file__, argparse.__file__]
    qs = ["Summarize what this code does in two sentences.",
          "List the public functions and one line on each.",
          "Find one possible bug or edge case and explain it.",
          "Write a docstring for the most complex function.",
          "What would you refactor first and why?",
          "Explain the control flow of the longest function.",
          "Which part is performance sensitive?",
          "Write two unit tests for the main entry point."]
    ps = []
    for f in files[:N_CTX]:
        src = open(f).read()[:CTX_TOKENS_CHARS]
        for q in qs[:N_Q]:   # first q per context = miss, rest = hits under a cache
            ps.append(f"Here is a source file:\n\n{src}\n\n---\nTask: {q}\nAnswer:")
    return ps

from vllm import LLM, SamplingParams
kw = dict(model=MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.80,
          max_num_seqs=1, seed=0, enforce_eager=False)
if arm == "r":
    kw["enable_prefix_caching"] = False
elif arm == "g":
    kw["enable_prefix_caching"] = True
elif arm in ("d", "s"):
    # LMCache in-process; vLLM's own prefix cache OFF so every hit comes from LMCache.
    # env (set by runner): d = CPU only, s = disk only (local_cpu False)
    from vllm.config import KVTransferConfig
    kw["enable_prefix_caching"] = False
    kw["gpu_memory_utilization"] = 0.80
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="LMCacheMPConnector", kv_role="kv_both",
        kv_connector_extra_config={"lmcache.mp.port": int(os.environ.get("MPPORT", 17555)),
            "lmcache.mp.heartbeat_interval": 60.0, "lmcache.mp.mq_timeout": 600.0})
else:
    raise SystemExit("arm not implemented: " + arm)
llm = LLM(**kw)
sp = SamplingParams(temperature=0, max_tokens=200, logprobs=0, seed=0)
res = []
for i, p in enumerate(prompts()):          # one request at a time = batch size 1
    ro = llm.generate([p], sp, use_tqdm=False)[0]
    o = ro.outputs[0]
    lp = [list(d.values())[0].logprob if d else None for d in o.logprobs]
    # chosen-token logprob: logprobs=0 returns just the sampled token's entry
    res.append({"i": i, "tokens": list(o.token_ids), "lp": lp, "cached": ro.num_cached_tokens,
                "ptoks": len(ro.prompt_token_ids)})
if arm in ("d", "s") and sum(x["cached"] or 0 for x in res if x["i"] % 8) == 0:
    json.dump({"arm": arm, "INVALID": "no cached tokens on hit prompts", "res": res}, open(out + ".INVALID", "w"))
    raise SystemExit("INVALID: LMCache served nothing (degraded mode?)")
json.dump({"arm": arm, "model": MODEL, "res": res}, open(out, "w"))
print("wrote", out, len(res))
