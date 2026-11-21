import json, os, itertools
R = {k: json.load(open(f"res/v23_{k}.json"))["res"] for k in ["r1","r2","g1","g2","d1","d2","s1","s2"]}
print("files INVALID:", [f for f in os.listdir("res") if "INVALID" in f])
print("\n== cached tokens (miss prompts i%8==0 | hit prompts)")
for k, r in R.items():
    miss = [x["cached"] or 0 for x in r if x["i"]%8==0]; hit = [x["cached"] or 0 for x in r if x["i"]%8]
    print(f"{k}: miss cached={miss}  hit cached>0={sum(1 for c in hit if c)}/35  distinct hit={sorted(set(hit))}  mean={sum(hit)/35:.0f} of ptoks mean {sum(x['ptoks'] for x in r)/40:.0f}")
def stat(a, b, idx):
    ident = bit = 0; first = []
    for i in idx:
        x, y = a[i], b[i]; n = min(len(x["tokens"]), len(y["tokens"]))
        d = next((k for k in range(n) if x["tokens"][k] != y["tokens"][k]), None)
        if d is None and len(x["tokens"]) == len(y["tokens"]):
            ident += 1; bit += x["lp"] == y["lp"]
        else: first.append(d if d is not None else n)
    return ident, bit, sorted(first)
miss = [i for i in range(40) if i%8==0]; hit = [i for i in range(40) if i%8]
print("\n== pairwise (token-identical / bitwise-logprob / diverged ; first-div positions)")
for a, b in [("r1","r2"),("g1","g2"),("d1","d2"),("s1","s2"),("r1","g1"),("r1","d1"),("r1","s1"),("g1","d1"),("g1","s1"),("d1","s1")]:
    for name, idx in (("miss", miss), ("hit", hit)):
        i, bt, f = stat(R[a], R[b], idx)
        print(f"{a} vs {b} [{name} n={len(idx)}]: ident={i} bitwise={bt} diverged={len(f)} first={f}")
