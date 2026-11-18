import sys, torch; from common import *
from sentence_transformers import SentenceTransformer
torch.set_num_threads(12)
models=sys.argv[1].split(','); dss=sys.argv[2].split(',')
for m in models:
    name,pre,kw=MODELS[m]
    todo=[d for d in dss if load(d) and not os.path.exists(f'{E}emb/{m}__{d}.npy')]
    if not todo: continue
    try:
        st=SentenceTransformer(name,device='cpu',**kw)
        for d in todo:
            t,_=load(d); t0=time.time()
            x=st.encode([pre+s for s in t],batch_size=64,normalize_embeddings=True,show_progress_bar=False)
            dt=time.time()-t0; np.save(f'{E}emb/{m}__{d}.npy',x.astype(np.float32))
            open(E+'encode_speed.jsonl','a').write(json.dumps(dict(model=m,ds=d,n=len(t),sec=dt,tps=len(t)/dt))+'\n'); print(m,d,len(t)/dt,flush=True)
    except Exception as e:
        print('FAIL',m,repr(e)[:300],flush=True); open(E+'fails.txt','a').write(f'{m}: {repr(e)[:300]}\n')
