# Alignment accuracy on the hand-labelled 2025 starts (truth table T below,
# each read off the transcript by hand), across onset-anchor settings.
# Needs the 2025 transcripts in MATCHWATCH_HOME; run with the project venv.
import json

from fgc_matchwatch import align, pipeline

H="/mnt/offload/fgc-matchwatch/"
api=json.load(open(H+"api2025.json")); names={r["team"]["country"]:r["team"]["name"] for r in api["rankings"]}
st=json.load(open(H+"streams-2025.json"))
# hand-verified truth (stream, key) -> true start, from the samples
T={("Hy2VGJjoMoo","t2-25"):3529,("Hy2VGJjoMoo","t2-13"):1848,("YRncWpIEcEQ","t2-50"):9190,("YRncWpIEcEQ","t2-20"):2588,
("XJjwMkiiwuQ","t2-95"):16574,("XJjwMkiiwuQ","t2-89"):15830,("FbAeVwfciMQ","t2-313"):4018,("FbAeVwfciMQ","t2-337"):7566,
("FbAeVwfciMQ","t2-355"):9879,("EMmcTueSVu0","t2-322"):5201,("EMmcTueSVu0","t2-346"):8789,("EMmcTueSVu0","t2-316"):4378,
("-fznHfw67Mg","t2-326"):5720,("-fznHfw67Mg","t2-344"):8172,("-fznHfw67Mg","t2-296"):1503,("W_ZVyhQoIOg","t2-329"):6093,
("W_ZVyhQoIOg","t2-305"):2634,("W_ZVyhQoIOg","t2-293"):1111,("W_ZVyhQoIOg","t2-359"):10291,("W_ZVyhQoIOg","t2-341"):7673,
("Js0dd0Aytj4","t4-3"):32632,("Js0dd0Aytj4","t2-312"):3609,("Js0dd0Aytj4","t2-351"):9199,("Js0dd0Aytj4","t2-309"):3161,
("Js0dd0Aytj4","t2-315"):3997,("t4IdgPIlFyg","t2-46"):7822,("-fznHfw67Mg","t2-332"):6534,("Js0dd0Aytj4","t2-297"):1537,
("bipuBmCye9g","t2-36"):4547,("bipuBmCye9g","t2-93"):12974,("EMmcTueSVu0","t2-340"):7807,("-fznHfw67Mg","t2-302"):2256,
("XJjwMkiiwuQ","t2-53"):10202,("bipuBmCye9g","t2-54"):6598,("-fznHfw67Mg","t2-362"):10711,("W_ZVyhQoIOg","t2-299"):1798,
("bipuBmCye9g","t2-66"):8283,("W_ZVyhQoIOg","t2-317"):4334,("bipuBmCye9g","t2-51"):6104,("Hy2VGJjoMoo","t2-109"):17552,
("t4IdgPIlFyg","t2-70"):11122,("Hy2VGJjoMoo","t2-97"):15763,("Hy2VGJjoMoo","t2-7"):1097,
# cross-stream disagreements resolved by reading (F5 = bipu, main = j2DN)
("bipuBmCye9g","t2-12"):1248,("bipuBmCye9g","t2-48"):6025,("j2DNyPMyKGE","t2-48"):10223,("bipuBmCye9g","t2-108"):15440,("j2DNyPMyKGE","t2-90"):12480+334+4189,
("j2DNyPMyKGE","t2-12"):1632,("j2DNyPMyKGE","t2-108"):19963,("j2DNyPMyKGE","t2-30"):4169,("bipuBmCye9g","t2-114"):16285,("bipuBmCye9g","t2-30"):3785,("j2DNyPMyKGE","t2-114"):20809}
cache={}
def run(vid):
    s=st[vid]; tr=json.load(open(H+f"transcripts/{vid}.json")); fld=s["field"] if s["field"] else 5
    same=[m for m in api["matches"] if m["field"]==fld and s["start"]-1800<=pipeline.match_time(m)<=s["start"]+tr["duration"]*1.6+3600]
    return {p.key:p for p in align.align(tr["words"],s["start"],[pipeline.spec_of(m) for m in same],names)}
vids=sorted({v for v,_ in T})
for on,back,fwd,pen in [(False,0,0,0),(True,160,400,0.75),(True,300,400,0.75),(True,400,600,0.75),(True,300,400,1.5)]:
    align.ONSET,align.ONSET_BACK,align.ONSET_FWD,align.ONSET_PENALTY=on,back,fwd,pen
    R={v:run(v) for v in vids}
    errs=[abs(R[v][k].start-t) if R[v][k].start is not None else 9999 for (v,k),t in T.items()]
    ok10=sum(e<=10 for e in errs); ok30=sum(e<=30 for e in errs)
    conf=sum(p.confident for v in vids for p in R[v].values()); n=sum(len(R[v]) for v in vids)
    print(f"onset={on} back={back} fwd={fwd} pen={pen}: within10s {ok10}/{len(T)} within30s {ok30}/{len(T)} confident {conf}/{n}  bad={[ (k,round(e)) for ((v,k),t),e in zip(T.items(),errs) if e>10]}")
