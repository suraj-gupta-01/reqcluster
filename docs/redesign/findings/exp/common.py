import torch
import os, json, time, numpy as np, pandas as pd
R='/home/hikigaya/Desktop/reqcluster/data/'
E='/tmp/claude-1000/-home-hikigaya-Desktop-reqcluster/0dfe9f9d-be6c-4b09-a5fe-c419290c1f2e/scratchpad/exp/'
DS={'uav2k':'uav_fms_2k','uav10k':'uav_fms_10k','promise':'promise','bit':'bit_control','multi':'multidomain','adv':'adversarial'}
MODELS={'minilm':('all-MiniLM-L6-v2','',{}),'bge-s':('BAAI/bge-small-en-v1.5','',{}),'bge-b':('BAAI/bge-base-en-v1.5','',{}),
'e5-b':('intfloat/e5-base-v2','query: ',{}),'mpnet':('sentence-transformers/all-mpnet-base-v2','',{}),
'mxbai':('mixedbread-ai/mxbai-embed-large-v1','',{'model_kwargs':{'torch_dtype':torch.float32}}),'gte-b':('Alibaba-NLP/gte-base-en-v1.5','',{'trust_remote_code':True,'model_kwargs':{'torch_dtype':torch.float32}}),
'nomic':('nomic-ai/nomic-embed-text-v1.5','clustering: ',{'trust_remote_code':True,'model_kwargs':{'torch_dtype':torch.float32}}),'qwen3':('Qwen/Qwen3-Embedding-0.6B','',{'model_kwargs':{'torch_dtype':torch.float32}})}
def load(ds):
    p=R+DS[ds]+'_requirements.csv'
    if not os.path.exists(p): return None
    d=pd.read_csv(p,keep_default_na=False); return d.text.fillna('').astype(str).tolist(), d.module.astype(str).values
