import json,sys
a=json.load(open(sys.argv[1]))["res"]; b=json.load(open(sys.argv[2]))["res"]
def stat(idx):
    ident=bit=0; first=[]
    for i in idx:
        x,y=a[i],b[i]; n=min(len(x["tokens"]),len(y["tokens"]))
        d=next((k for k in range(n) if x["tokens"][k]!=y["tokens"][k]),None)
        if d is None and len(x["tokens"])==len(y["tokens"]):
            ident+=1; bit+= x["lp"]==y["lp"]
        else: first.append(d if d is not None else n)
    return f"n={len(idx)} token-identical={ident} bitwise={bit} diverged={len(first)} first-div={sorted(first)}"
miss=[i for i in range(40) if i%8==0]; hit=[i for i in range(40) if i%8!=0]
print("MISS (G computes like R):", stat(miss))
print("HIT  (G reuses cached KV):", stat(hit))
