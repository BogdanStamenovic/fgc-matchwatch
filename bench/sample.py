# Print transcript around placed starts for a seeded random sample (hand check).
import json
import random
import subprocess
import sys

H="/mnt/offload/fgc-matchwatch/"
api=json.load(open(H+"api2025.json")); names={r["team"]["country"]:r["team"]["name"] for r in api["rankings"]}
mm={f'{m["tournamentKey"]}-{m["id"]}':m for m in api["matches"]}
random.seed(int(sys.argv[1])); K=int(__import__("os").environ.get("K","2"))
for vid in sys.argv[2:]:
    r=json.loads(subprocess.run([H.replace("/mnt/offload/fgc-matchwatch/","")+"/home/bodas/data/fgc-matchwatch/.venv/bin/fgc-matchwatch","-q","--year","2025","align","--",vid],capture_output=True,text=True).stdout)
    tr=json.load(open(H+f"transcripts/{vid}.json"))
    for p in random.sample([x for x in r if x["start"] and x["confident"]],K):
        m=mm[p["key"]]
        red=[names.get(x["country"],x["country"]) for x in m["participants"] if x["station"]<20]; blue=[names.get(x["country"],x["country"]) for x in m["participants"] if x["station"]>20]
        print(f"=== {vid} {p['key']} placed={p['start']:.0f} score={m['redScore']}-{m['blueScore']} RED={red} BLUE={blue}")
        a=p["start"]-25; b=p["start"]+12
        ws=[w for w in tr["words"] if a<=w[0]<=b]
        print(" ".join(f"[{w[0]-p['start']:+.0f}]{w[2]}" if i%6==0 else w[2] for i,w in enumerate(ws)))
        ws=[w for w in tr["words"] if p["start"]+150<=w[0]<=p["start"]+330]
        print("  AFTER:", " ".join(w[2] for w in ws)[:160])
