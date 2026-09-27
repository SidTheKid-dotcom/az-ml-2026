"""Consensus France file: S1-file French rows, minus links all 4 string models remove, plus links all 4 add
that pass the guard (in v2_guard's accepted set). One owner and cap 11 enforced; non-French lines byte-identical."""
import sys, hashlib, json; sys.path.insert(0,'/home/ubuntu/run27/G/code')
from pathlib import Path
from ber.string_gbdt_apply import read_links
G='/home/ubuntu/run27/G/'
vs=['v1','v2','v2_us','fr_v3_hybrid']
fr=[]
with open(G+'deliver/v1/france_rows.tsv') as f:
    next(f)
    for line in f: fr.append(line.split('\t',1)[0])
frs=set(fr)
base=read_links(G+'data/s1_matching_results.tsv', frs)
L={v:read_links(G+f'deliver/{v}/france_rows.tsv') for v in vs}
guarded=read_links(G+'deliver/v2_guard/france_rows.tsv') | read_links(G+'deliver/v1_guard/france_rows.tsv')
addA=set.intersection(*[L[v]-base for v in vs]); remR=set.intersection(*[base-L[v] for v in vs])
addA_g={p for p in addA if p in guarded}
# base lists in order
lists={}
with open(G+'data/s1_matching_results.tsv') as f:
    next(f)
    for line in f:
        sid,_,r=line.rstrip('\n').partition('\t')
        if sid in frs: lists[sid]=[x for x in r.split(',') if x] if r else []
owner={}
for s,rs in lists.items():
    lists[s]=[r for r in rs if (s,r) not in remR]
    for r in lists[s]: owner[r]=s
skipped=0
# model prob for ordering adds
import pandas as pd
ps=pd.read_parquet(G+'deliver/v2/pair_scores.parquet', columns=['s1_id','rec_id','prob'])
ps=ps[ps.s1_id.isin({s for s,_ in addA_g})]
pr=dict(zip(zip(ps.s1_id,ps.rec_id),ps.prob))
for s,r in sorted(addA_g, key=lambda p:-pr.get(p,0)):
    if r in owner or len(lists[s])>=11: skipped+=1; continue
    lists[s].append(r); owner[r]=s
out=Path(G+'deliver/consensus4_guard'); (out/'upload_france_only').mkdir(parents=True, exist_ok=True)
with open(out/'france_rows.tsv','w') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for s in fr: f.write(s+'\t'+','.join(lists[s])+'\n')
n=0; seen=set(); allown={}
with open(G+'data/s1_matching_results.tsv') as fin, open(out/'upload_france_only/matching_results.tsv','w') as fo:
    fo.write(next(fin))
    for line in fin:
        sid=line.split('\t',1)[0]; assert sid not in seen; seen.add(sid); n+=1
        if sid in lists: line=sid+'\t'+','.join(lists[sid])+'\n'
        fo.write(line)
        rs=line.rstrip('\n').split('\t')[1]
        rs=rs.split(',') if rs else []
        assert len(rs)<=11 and len(set(rs))==len(rs)
        for r in rs: assert r not in allown, (r, sid); allown[r]=sid
assert n==1732544
data=open(out/'upload_france_only/matching_results.tsv','rb').read(); assert b'\r' not in data
new=sum(len(v) for v in lists.values())
h=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
info={'links':new,'base':len(base),'added_all4':len(addA),'added_all4_guarded':len(addA_g),'removed_all4':len(remR),'add_skipped_owner_or_cap':skipped,
      'sha256':{'france_rows':h(out/'france_rows.tsv'),'upload':h(out/'upload_france_only/matching_results.tsv')}}
json.dump(info, open(out/'info.json','w'), indent=1); print(info)
