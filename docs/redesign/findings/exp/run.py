import os,sys
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']: os.environ[v]='1'
os.environ['NUMBA_CACHE_DIR']='/tmp/numba_exp'
import warnings; warnings.filterwarnings('ignore')
from common import *
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, silhouette_score
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster

def metrics(gold,pred):
    gold=np.asarray(gold); pred=np.asarray(pred); n=len(gold)
    ari=ARI(gold,pred); nmi=NMI(gold,pred)
    isn=gold=='__noise__'; gl=sorted(set(gold[~isn])); pl=sorted(set(pred[pred>=0]))
    ok=int(np.sum(isn&(pred<0)))
    if pl:
        M=np.zeros((len(gl),len(pl)),int)
        gi={g:i for i,g in enumerate(gl)}; pi={p:i for i,p in enumerate(pl)}
        for g,p in zip(gold[~isn],pred[~isn]):
            if p>=0: M[gi[g],pi[p]]+=1
        r,c=linear_sum_assignment(-M); ok+=int(M[r,c].sum())
    return dict(ari=ari,nmi=nmi,acc=ok/n,noise=float(np.mean(pred<0)),k=len(pl),kgold=len(gl))

def reduce(X,spec,seed):
    n=len(X); t=time.time()
    kind=spec.split('_')[0]
    if kind=='umap':
        import umap
        nn={'15':15,'30':30,'S':int(min(80,max(15,n//100))),'C':int(min(80,max(15,n//500)))}[spec.split('_')[1]]
        init='spectral' if spec.endswith('_sp') else 'random'
        Z=PCA(50,random_state=seed).fit_transform(X)
        Z=umap.UMAP(n_components=10,metric='cosine',n_neighbors=nn,min_dist=0.0,random_state=seed,init=init,n_jobs=1).fit_transform(Z)
    elif kind=='pca':
        Z=PCA(int(spec.split('_')[1]),random_state=seed).fit_transform(X); Z/=np.linalg.norm(Z,axis=1,keepdims=True)+1e-9
    else: Z=X
    return Z,time.time()-t

def clusterers(n):
    out=[]
    mcs={'n50':max(5,n//50),'n100':max(5,n//100),'sq':max(5,int(n**.5))}
    for a,m in mcs.items():
        for ms in (1,3,5):
            for sel in ('eom','leaf'): out.append((f'hdb_{a}_s{ms}_{sel}',m,ms,sel))
    return out

def task(a):
    m,ds,spec,seed,ref=a
    X=np.load(f'{E}emb/{m}__{ds}.npy'); gold=load(ds)[1]; n=len(X); rows=[]
    Z,tr=reduce(X,spec,seed)
    base=dict(model=m,ds=ds,red=spec,seed=seed,t_reduce=tr,n=n)
    for name,mcs,ms,sel in clusterers(n):
        t=time.time(); lab=HDBSCAN(min_cluster_size=mcs,min_samples=min(ms,mcs-1),cluster_selection_method=sel).fit(Z).labels_
        rows.append({**base,'clu':name,'t_cluster':time.time()-t,**metrics(gold,lab)})
    if ref:
        ks=[k for k in (4,6,8,10,12,14,16,18,20,24,28,32,40) if k<n//4]; ss=min(n,3000)
        best=None
        for k in ks:
            t=time.time(); lab=KMeans(k,n_init=3,random_state=seed).fit_predict(Z)
            s=silhouette_score(Z,lab,sample_size=ss,random_state=0)
            if best is None or s>best[0]: best=(s,lab,k,time.time()-t)
        rows.append({**base,'clu':'kmeans_sil','t_cluster':best[3],**metrics(gold,best[1])})
        if n<=10000:
            for nm,meth,met in (('ward','ward','euclidean'),('avgcos','average','cosine')):
                t=time.time(); L=linkage(Z,meth,metric=met) if meth!='ward' else linkage(Z,'ward'); tl=time.time()-t; best=None
                for k in ks:
                    lab=fcluster(L,k,'maxclust')-1
                    s=silhouette_score(Z,lab,metric=met if meth!='ward' else 'euclidean',sample_size=ss,random_state=0)
                    if best is None or s>best[0]: best=(s,lab)
                rows.append({**base,'clu':'agglo_'+nm,'t_cluster':tl,**metrics(gold,best[1])})
    return rows

if __name__=='__main__':
    models=sys.argv[1].split(','); dss=sys.argv[2].split(','); specs=sys.argv[3].split(','); seeds=int(sys.argv[4]); out=sys.argv[5]; ref=len(sys.argv)>6
    tasks=[]
    for m in models:
        for ds in dss:
            if not os.path.exists(f'{E}emb/{m}__{ds}.npy'): continue
            for sp in specs:
                if sp=='umap_C' and ds!='uav10k': continue
                for s in (range(seeds) if sp.startswith('umap') else [0]): tasks.append((m,ds,sp,s,ref and sp in ('umap_15','umap_S','umap_C','pca_20','raw') and s==0))
    print(len(tasks),'tasks',flush=True)
    with Pool(int(os.environ.get('NP',10)),maxtasksperchild=1) as p, open(E+out,'a') as f:
        for rows in p.imap_unordered(task,tasks):
            for r in rows: f.write(json.dumps(r)+'\n')
            f.flush()
