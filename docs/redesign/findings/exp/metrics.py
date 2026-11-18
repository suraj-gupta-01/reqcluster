import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from scipy.optimize import linear_sum_assignment
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
