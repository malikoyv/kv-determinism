"""Compare two gen.py outputs: identical sequences, first divergence, logprob deltas."""
import sys, json
a, b = (json.load(open(p))["res"] for p in sys.argv[1:3])
same = 0; first = []; maxd = 0.0; bit = 0
for x, y in zip(a, b):
    n = min(len(x["tokens"]), len(y["tokens"]))
    d = next((k for k in range(n) if x["tokens"][k] != y["tokens"][k]), None)
    if d is None and len(x["tokens"]) == len(y["tokens"]):
        same += 1
        if x["lp"] == y["lp"]: bit += 1
        else: maxd = max(maxd, max(abs(p - q) for p, q in zip(x["lp"], y["lp"])))
    else:
        first.append(d if d is not None else n)
print(f"prompts {len(a)}  identical tokens {same}  bitwise-identical logprobs {bit}  diverged {len(first)}")
print("first-divergence positions:", sorted(first))
print("max |dlogprob| among token-identical:", maxd)
