import json, itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ks = {"g":"g1","d64":"dc64","d128":"dc128","d256":"d1","d512":"dc512","s256":"s1"}
R = {k: json.load(open(f"res/v23_{v}.json"))["res"] for k, v in ks.items()}
hit = [i for i in range(40) if i % 8]
same = lambda x, y: x["tokens"] == y["tokens"] and x["lp"] == y["lp"]
eq = [0, 0]; ne = [0, 0]
for a, b in itertools.combinations(ks, 2):
    for i in hit:
        x, y = R[a][i], R[b][i]; t = eq if x["cached"] == y["cached"] else ne
        t[0] += same(x, y); t[1] += 1
ds = [sum(same(R["d256"][i], R["s256"][i]) for i in hit), len(hit)]
rows = [("Same number of reused tokens\n(any tier, any chunk size)", eq),
        ("Different number of reused tokens", ne),
        ("DRAM vs SSD, same chunk size\n(a subset of the first row)", ds)]
INK, INK2, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100, facecolor=SURF); ax.set_facecolor(SURF)
for k, (lab, (s, n)) in enumerate(rows):
    y = len(rows) - 1 - k
    ax.barh(y, s / n, height=0.46, color=BLUE, edgecolor=SURF, linewidth=2)
    ax.barh(y, 1 - s / n, left=s / n, height=0.46, color=ORANGE, edgecolor=SURF, linewidth=2)
    ax.text(-0.02, y, lab, ha="right", va="center", fontsize=13, color=INK)
    ax.text(1.02, y, f"{s} of {n}\nidentical", ha="left", va="center", fontsize=14, color=INK, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(-0.6, 2.6); ax.axis("off")
fig.text(0.04, 0.93, "After a cache hit, the answer depends on how many tokens were reused,", fontsize=19, color=INK, fontweight="bold")
fig.text(0.04, 0.875, "not on where the KV cache was stored.", fontsize=19, color=INK, fontweight="bold")
fig.text(0.04, 0.815, "Share of runs whose output is bit-identical (tokens and logprobs), greedy decoding, batch size 1", fontsize=12.5, color=INK2)
fig.patches += [plt.Rectangle((0.04, 0.755), 0.014, 0.022, transform=fig.transFigure, color=BLUE), plt.Rectangle((0.255, 0.755), 0.014, 0.022, transform=fig.transFigure, color=ORANGE)]
fig.text(0.06, 0.757, "bit-identical output", fontsize=12.5, color=INK2); fig.text(0.275, 0.757, "different output", fontsize=12.5, color=INK2)
fig.text(0.04, 0.085, "Qwen2.5-3B, bf16, one RTX 5070 Ti, vLLM 0.23 + LMCache 0.5.7. 35 cache-hit prompts from 5 contexts, every pair of 6 runs\n"
         "(vLLM GPU cache; LMCache DRAM at chunk 64/128/256/512; LMCache SSD at 256).\n"
         "Counts are prompt comparisons, not independent samples. One model, one GPU.",
         fontsize=10.5, color=INK2, va="center")
fig.subplots_adjust(left=0.33, right=0.84, top=0.70, bottom=0.17)
fig.savefig("fig_reused_length.png", facecolor=SURF)
print("same:", eq, "different:", ne, "dram-vs-ssd:", ds)
