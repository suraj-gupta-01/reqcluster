import os,sys,warnings; warnings.filterwarnings('ignore')
os.environ['NUMBA_CACHE_DIR']='/tmp/numba_exp'; os.environ['OMP_NUM_THREADS']='2'
from common import *; from metrics import metrics
from sklearn.decomposition import PCA; from sklearn.cluster import HDBSCAN; import umap
from scipy.cluster.hierarchy import linkage,fcluster; from sklearn.metrics import silhouette_score
seed=int(sys.argv[1]); X=np.load(E+'emb/minilm__uav10k.npy'); gold=load('uav10k')[1]; n=len(X)
def umapz(nn,ep,Z): return umap.UMAP(n_components=10,metric='cosine',n_neighbors=nn,min_dist=0.0,random_state=seed,init='random',n_epochs=ep,n_jobs=1).fit_transform(Z)
def hdb(Z): return HDBSCAN(min_cluster_size=max(5,n//50),min_samples=3).fit(Z).labels_
def agg(Z):
    L=linkage(Z,'ward'); best=None
    for k in (8,10,12,14,16,18,20,24,28,32,40):
        l=fcluster(L,k,'maxclust')-1; s=silhouette_score(Z,l,sample_size=2000,random_state=0)
        if best is None or s>best[0]: best=(s,l)
    return best[1]
for name,nn,ep,cl in (('cur_nn20_ep500_hdb',20,None,hdb),('nn80_ep500_hdb',80,None,hdb),('nn80_ep200_hdb',80,200,hdb),('nn80_ep200_wardsil',80,200,agg),('nn20_ep200_hdb',20,200,hdb),('nn50_ep200_hdb',50,200,hdb),('nn50_ep200_wardsil',50,200,agg)):
    t0=time.time(); Zp=PCA(50,random_state=seed).fit_transform(X); t1=time.time(); Z=umapz(nn,ep,Zp); t2=time.time(); lab=cl(Z); t3=time.time()
    r=dict(cfg=name,seed=seed,t_pca=t1-t0,t_umap=t2-t1,t_clu=t3-t2,**metrics(gold,lab)); print(json.dumps(r),flush=True); open(E+'t10k.jsonl','a').write(json.dumps(r)+'\n')
