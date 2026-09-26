import numpy as np, pyarrow.parquet as pq, time
t=time.time()
tab=pq.read_table("/workspace/az-ml-new/arena/v1/pairs.parquet",columns=["left_idx","right_idx"])
print(tab.schema)
left=tab.column("left_idx").to_numpy(); right=tab.column("right_idx").to_numpy()
print(len(left), left.dtype, time.time()-t, "sorted left", bool(np.all(left[1:]>=left[:-1])))
part=np.load("/workspace/az-ml-new/arena/finalize_v1/s1_partition.npy"); comp=np.load("/workspace/az-ml-new/arena/finalize_v1/component_ids.npy")
print("part", part.dtype, np.bincount(part), "comp", comp.dtype, comp.max())
# does partition vary inside component?
o=np.lexsort((part,comp)); c=comp[o]; p=part[o]
st=np.r_[True,c[1:]!=c[:-1]]; g=np.cumsum(st)-1
pmin=np.minimum.reduceat(p,np.flatnonzero(st)); pmax=np.maximum.reduceat(p,np.flatnonzero(st))
print("components with mixed partitions", int((pmin!=pmax).sum()))
base=np.load("/workspace/az-ml-new/arena/v1/base_probabilities.npy",mmap_mode="r")
blend=np.load("/workspace/az-ml-new/arena/v1/probabilities.npy",mmap_mode="r")
gated=np.asarray(base)>=0.01
mask=np.load("/opt/az-ml-new/runs/c2_step005000/b_rest_scores/scored_mask.npy")
brest=part[left]==1
print("gated", int(gated.sum()), "brest gated", int((gated&brest).sum()), "mask", int(mask.sum()), "mask==gated&brest", bool(np.array_equal(mask, gated&brest)))
n_rec=int(right.max())+1
rec_brest=np.bincount(right[gated&brest],minlength=n_rec)>0
rec_other=np.bincount(right[gated&~brest],minlength=n_rec)>0
print("records gated by brest", int(rec_brest.sum()), "also gated by non-brest", int((rec_brest&rec_other).sum()))
strong=np.asarray(blend)>=0.5
rb=np.bincount(right[strong&brest],minlength=n_rec)>0; ro=np.bincount(right[strong&~brest],minlength=n_rec)>0
print("records blend>=.5 by brest", int(rb.sum()), "also by non-brest >=.5", int((rb&ro).sum()), "brest-strong recs with any gated non-brest claimant", int((rb&rec_other).sum()))
# brest gated rows whose record has a non-brest gated claimant
print("brest gated rows w/ outside rival", int((gated&brest&rec_other[right]).sum()))
lab=np.load("/workspace/az-ml-new/arena/labels/label.npy",mmap_mode="r")
print("time", time.time()-t)
