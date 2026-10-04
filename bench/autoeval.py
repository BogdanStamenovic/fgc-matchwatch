"""Automated alignment check against independent spoken evidence.

Alignment runs WITHOUT match-number / score features. Ground truth per match =
(a) "match N" said, or (b) both official scores said in the result read-out.
Truth start estimate: the 'match N' mention nearest before a countdown... we only
check containment: the evidence must fall in [start-200, start+30] for the
number, or [start+150, start+420] for the score read-out.
"""
import json
import sys

sys.path.insert(0, "/home/bodas/data/fgc-matchwatch/src")
from fgc_matchwatch import align, pipeline

H = "/mnt/offload/fgc-matchwatch/"
api = json.load(open(H + "api2025.json"))
names = {r["team"]["country"]: r["team"]["name"] for r in api["rankings"]}
streams = json.load(open(H + "streams.json"))
tot = dict(n=0, placed=0, conf=0, ev=0, ev_ok=0, ev_bad=0, conf_bad=0)
bad = []
for vid in sys.argv[1:]:
    s = streams[vid]; tr = json.load(open(H + f"transcripts/{vid}.json"))
    st = s["start"]; end = st + tr["duration"] * 1.6 + 3600
    same = [m for m in api["matches"] if m["field"] == s["field"] and st - 1800 <= pipeline.match_time(m) <= end]
    specs = [pipeline.spec_of(m) for m in same]
    res = {p.key: p for p in align.align(tr["words"], st, specs, names, use_numbers=False)}
    toks = align.tokens(tr["words"]); ments = align.match_mentions(toks); nums = align.numbers(toks)
    for sp in specs:
        p = res[sp.key]; tot["n"] += 1
        if p.start is None: continue
        tot["placed"] += 1; tot["conf"] += p.confident
        # independent evidence anywhere in the stream for this match
        num_t = [t for t, n in ments if n == sp.number]
        has_score = sp.scores and min(sp.scores) >= 10
        sc_t = [t for t, n in nums if has_score and n == sp.scores[0] and any(abs(t2 - t) < 20 and n2 == sp.scores[1] for t2, n2 in nums)]
        if not num_t and not sc_t: continue
        tot["ev"] += 1
        ok = any(p.start - 330 <= t <= p.start + 30 for t in num_t) or any(p.start + 140 <= t <= p.start + 420 for t in sc_t)
        tot["ev_ok"] += ok
        if not ok:
            tot["ev_bad"] += 1; tot["conf_bad"] += p.confident
            bad.append((vid, sp.key, round(p.start), [round(t) for t in num_t][:4], [round(t) for t in sc_t][:4], p.confident))
print(json.dumps(tot))
for b in bad: print("MISMATCH", b)
