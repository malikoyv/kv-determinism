import json, itertools
ks = {"g":"g1","d64":"dc64","d128":"dc128","d256":"d1","d512":"dc512","s256":"s1"}
R = {k: json.load(open(f"res/v23_{v}.json"))["res"] for k, v in ks.items()}
hit = [i for i in range(40) if i % 8]
def same(x, y): return x["tokens"] == y["tokens"] and x["lp"] == y["lp"]
tot = {"equal": [0, 0], "unequal": [0, 0]}
rows = []
for a, b in itertools.combinations(ks, 2):
    e = [0, 0]; u = [0, 0]
    for i in hit:
        x, y = R[a][i], R[b][i]
        t = e if x["cached"] == y["cached"] else u
        t[0] += same(x, y); t[1] += 1
    rows.append((a, b, e, u))
    for n, t in (("equal", e), ("unequal", u)):
        tot[n][0] += t[0]; tot[n][1] += t[1]
for a, b, e, u in rows:
    print(f"{a:>5} vs {b:<5} same reused length: bit-identical {e[0]}/{e[1]} | different length: bit-identical {u[0]}/{u[1]}")
print("\nALL PAIRS, hit prompts: same reused length -> bit-identical %d/%d ; different length -> bit-identical %d/%d" % (tot["equal"][0], tot["equal"][1], tot["unequal"][0], tot["unequal"][1]))
