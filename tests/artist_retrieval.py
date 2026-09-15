import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import sqlite3, json
import numpy as np
from timbre.groundtruth import load_layout
from timbre.search import exact_track_topk

DB,ST="store/timbre.db","store/vectors.npy"
row_ids,owner=load_layout(DB)
store=np.load(ST,mmap_mode="r")
c=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
rows=c.execute("""select track_id,row_start,n_windows,title,artist,genre from tracks
                  where status='done' and n_windows>0""").fetchall()
T={r[0]:r for r in rows}
by_artist={}
for r in rows:
    if r[4]: by_artist.setdefault(r[4],[]).append(r[0])

# Artists with >=20 tracks: enough siblings that missing them all is meaningful.
cands=sorted([(a,t) for a,t in by_artist.items() if len(t)>=20], key=lambda x:-len(x[1]))
rng=np.random.default_rng(7)
K=20
print(f"{len(cands)} artists with >=20 tracks in the medium store\n")
print(f"{'artist':<26}{'trks':>5}{'hits@20':>9}{'exp':>7}{'lift':>7}  {'genre':<14}")
out=[]
for artist,tracks in cands[:14]:
    share=len(tracks)/len(rows)          # expected hits if retrieval were random
    hits=[]
    picks=rng.choice(tracks,size=min(5,len(tracks)),replace=False)
    for tid in picks:
        t=T[tid]; w=min(t[2]//2, t[2]-1)         # mid-track window
        q=np.asarray(store[t[1]+w])
        ids,_,_=exact_track_topk(store,q,row_ids,owner,k=K,exclude_track=tid)
        hits.append(sum(1 for i in ids if T[int(i)][4]==artist))
    mean=float(np.mean(hits)); exp=share*K
    out.append({"artist":artist,"n_tracks":len(tracks),"mean_hits_at_20":mean,
                "expected":exp,"lift":mean/exp if exp else 0,
                "genre":T[tracks[0]][5],"per_query":hits,"queried":[int(p) for p in picks]})
    print(f"{artist[:25]:<26}{len(tracks):>5}{mean:>9.1f}{exp:>7.2f}{mean/exp if exp else 0:>7.1f}x  {str(T[tracks[0]][5])[:13]:<14}")
json.dump(out,open("docs/listening_artist_retrieval.json","w"),indent=2)
m=[o["lift"] for o in out]
print(f"\nmedian lift over chance: {np.median(m):.1f}x")
