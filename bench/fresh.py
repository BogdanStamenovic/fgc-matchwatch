# Draw a random sample of placed matches not used for tuning and print the
# transcript around each placed start, for checking by hand.
import json
import random
import subprocess

H="/mnt/offload/fgc-matchwatch/"
seen=set(["t2-25", "t2-13", "t2-50", "t2-20", "t2-95", "t2-89", "t2-313", "t2-337", "t2-355", "t2-322", "t2-346", "t2-316", "t2-326", "t2-344", "t2-296", "t2-329", "t2-305", "t2-293", "t2-359", "t2-341", "t2-330", "t4-3", "t2-312", "t2-351", "t2-309", "t2-315", "t2-46", "t2-7", "t2-97", "t2-43", "t2-98", "t2-332", "t2-297", "t2-36", "t2-93", "t2-340", "t2-302", "t2-53", "t2-54", "t2-362", "t2-299", "t2-66", "t2-317", "t2-51", "t2-109", "t2-70", "t2-12", "t2-48", "t2-108", "t2-90", "t2-75"])
api=json.load(open(H+"api2025.json")); names={r["team"]["country"]:r["team"]["name"] for r in api["rankings"]}
mm={f'{m["tournamentKey"]}-{m["id"]}':m for m in api["matches"]}
pool=[]
for vid in ["Hy2VGJjoMoo", "t4IdgPIlFyg", "YRncWpIEcEQ", "XJjwMkiiwuQ", "bipuBmCye9g", "FbAeVwfciMQ", "EMmcTueSVu0", "-fznHfw67Mg", "W_ZVyhQoIOg", "Js0dd0Aytj4"]:
    r=json.loads(subprocess.run(["/home/bodas/data/fgc-matchwatch/.venv/bin/fgc-matchwatch","-q","--year","2025","align","--",vid],capture_output=True,text=True).stdout)
    pool+= [(vid,p) for p in r if p["key"] not in seen]
random.seed(31337)
print("pool",len(pool),"placed",sum(1 for _,p in pool if p["start"]),"confident",sum(1 for _,p in pool if p["confident"]))
for vid,p in random.sample(pool,15):
    m=mm[p["key"]]; tr=json.load(open(H+f"transcripts/{vid}.json"))
    red=[names.get(x["country"],x["country"]) for x in m["participants"] if x["station"]<20]; blue=[names.get(x["country"],x["country"]) for x in m["participants"] if x["station"]>20]
    print(f"\n=== {vid} {p['key']} placed={p['start'] and round(p['start'])} conf={p['confident']} kind={p['anchor_kind']} RED={red} BLUE={blue}")
    if not p["start"]: continue
    ws=[w for w in tr["words"] if p["start"]-40<=w[0]<=p["start"]+25]
    print(" ".join(f"[{w[0]-p['start']:+.0f}]{w[2]}" if i%5==0 else w[2] for i,w in enumerate(ws))[:900])
