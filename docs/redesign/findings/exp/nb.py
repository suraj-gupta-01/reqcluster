import os,sys,time; os.environ['NUMBA_CACHE_DIR']=sys.argv[1]
import numpy as np; t=time.time(); import umap; ti=time.time()-t
X=np.load('emb/minilm__uav2k.npy'); t=time.time()
umap.UMAP(n_components=10,metric='cosine',n_neighbors=15,min_dist=0.0,random_state=0,init='random',n_epochs=200,n_jobs=1).fit_transform(X[:,:50] if False else X); print(sys.argv[1],'import',round(ti,1),'fit',round(time.time()-t,1))
