import json, os
ks = {"r1":"r1","d256":"d1","d64":"dc64","d128":"dc128","d512":"dc512","g":"g1"}
R = {k: json.load(open(f"res/v23_{v}.json"))["res"] for k, v in ks.items()}
print("INVALID files:", [f for f in os.listdir("res") if "INVALID" in f and "d1_degraded" not in f])
hit = [i for i in range(40) if i % 8]; miss = [i for i in range(40) if i % 8 == 0]
print("\n== cached tokens on hits (distinct) and multiple-of-chunk check")
for k, c in (("d64",64),("d128",128),("d256",256),("d512",512),("g",16)):
    h = [R[k][i]["cached"] or 0 for i in hit]
    print(f"{k}: distinct={sorted(set(h))} mean={sum(h)/35:.0f} all multiples of {c}: {all(x % c == 0 for x in h)} hits>0: {sum(1 for x in h if x)}/35")
def stat(a, b, idx):
    ident = bit = 0; first = []
    for i in idx:
        x, y = R[a][i], R[b][i]; n = min(len(x["tokens"]), len(y["tokens"]))
        d = next((k for k in range(n) if x["tokens"][k] != y["tokens"][k]), None)
        if d is None and len(x["tokens"]) == len(y["tokens"]):
            ident += 1; bit += x["lp"] == y["lp"]
        else: first.append(d if d is not None else n)
    return f"ident={ident} bitwise={bit} diverged={len(first)}"
print("\n== pairwise on HIT prompts (n=35) | MISS prompts (n=5)")
for a, b in [("r1","d64"),("r1","d128"),("r1","d256"),("r1","d512"),("r1","g"),("d64","d128"),("d128","d256"),("d256","d512"),("d64","d512"),("g","d64"),("g","d128"),("g","d256"),("g","d512")]:
    print(f"{a:>5} vs {b:<5} hit: {stat(a,b,hit)} | miss: {stat(a,b,miss)}")
