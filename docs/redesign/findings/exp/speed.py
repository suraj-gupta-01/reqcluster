import os,sys,warnings; warnings.filterwarnings('ignore')
os.environ['NUMBA_CACHE_DIR']='/tmp/numba_exp'
from common import *
import torch; torch.set_num_threads(12)
from sentence_transformers import SentenceTransformer
from sentence_transformers.backend import export_dynamic_quantized_onnx_model
from sklearn.decomposition import PCA; from sklearn.cluster import HDBSCAN; import umap
sys.argv+= [] ; 
from metrics import metrics
key=sys.argv[1]; ds=sys.argv[2]; name,pre,kw=MODELS[key]; nmax=int(sys.argv[3]) if len(sys.argv)>3 else 10**9
t,gold=load(ds); t=t[:nmax]; gold=gold[:nmax]; pre_t=[pre+s for s in t]
def evalemb(X,tag,extra={},epochs=None):
    out=[]
    for s in range(3):
        t0=time.time(); Z=PCA(50,random_state=s).fit_transform(X)
        Z=umap.UMAP(n_components=10,metric='cosine',n_neighbors=int(min(80,max(15,len(X)//500))),min_dist=0.0,random_state=s,init='random',n_epochs=epochs,n_jobs=1).fit_transform(Z); tu=time.time()-t0
        lab=HDBSCAN(min_cluster_size=max(5,len(X)//50),min_samples=3).fit(Z).labels_; out.append({**metrics(gold,lab),'t_umap':tu})
    d=pd.DataFrame(out).mean(); r=dict(tag=tag,ari=d.ari,acc=d.acc,nmi=d.nmi,noise=d.noise,t_umap=d.t_umap,**extra); print(json.dumps(r),flush=True); open(E+'speed.jsonl','a').write(json.dumps(r)+'\n')
def enc(st,tag):
    st.encode(pre_t[:64],batch_size=64)  # warmup
    t0=time.time(); X=st.encode(pre_t,batch_size=64,normalize_embeddings=True); dt=time.time()-t0
    return X,dict(model=key,ds=ds,n=len(t),enc_sec=dt,tps=len(t)/dt)
ref=None
base=SentenceTransformer(name,device='cpu',**kw); print('maxseq',base.max_seq_length)
X,ex=enc(base,'torch'); ref=X; evalemb(X,'torch_default',ex)
evalemb(X,'torch_default_epochs200',ex,epochs=200)
base.max_seq_length=128; X,ex=enc(base,'t128'); evalemb(X,'torch_seq128',{**ex,'cos_vs_ref':float((X*ref).sum(1).mean())})
base.max_seq_length=64; X,ex=enc(base,'t64'); evalemb(X,'torch_seq64',{**ex,'cos_vs_ref':float((X*ref).sum(1).mean())})
try:
    ox=SentenceTransformer(name,device='cpu',backend='onnx',**kw); X,ex=enc(ox,'onnx'); evalemb(X,'onnx_fp32',{**ex,'cos_vs_ref':float((X*ref).sum(1).mean())})
    d=E+f'onnxq_{key}'
    export_dynamic_quantized_onnx_model(ox,'avx512_vnni' if 'avx512_vnni' in open('/proc/cpuinfo').read() else 'avx2',d)
    import glob; f=[x for x in glob.glob(d+'/**/*.onnx',recursive=True)]; print(f)
    q=SentenceTransformer(d,device='cpu',backend='onnx',model_kwargs={'file_name':os.path.relpath(f[0],d)},**kw) if f else None
    X,ex=enc(q,'int8'); evalemb(X,'onnx_int8',{**ex,'cos_vs_ref':float((X*ref).sum(1).mean())})
    q.max_seq_length=128; X,ex=enc(q,'int8_128'); evalemb(X,'onnx_int8_seq128',{**ex,'cos_vs_ref':float((X*ref).sum(1).mean())})
except Exception as e: print('ONNX FAIL',repr(e)[:400]); open(E+'fails.txt','a').write(f'onnx {key}: {repr(e)[:400]}\n')
