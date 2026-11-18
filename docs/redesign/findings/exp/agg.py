import pandas as pd,sys,glob
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',200)
d=pd.concat([pd.read_json(f,lines=True) for f in sys.argv[1].split(',')])
dsl=sys.argv[2].split(',') ; top=int(sys.argv[3]) if len(sys.argv)>3 else 15
g=d.groupby(['model','red','clu','ds'])[['ari','nmi','acc','noise','k','kgold','t_reduce','t_cluster']].mean().reset_index()
sd=d.groupby(['model','red','clu','ds'])['ari'].std().fillna(0).rename('ari_sd').reset_index()
g=g.merge(sd)
p=g.pivot_table(index=['model','red','clu'],columns='ds',values=['ari','acc'])
p=p[[c for c in p.columns if c[1] in dsl]].dropna()
p['ARI']=p['ari'].mean(axis=1); p['ACC']=p['acc'].mean(axis=1); p['S']=(p.ARI+p.ACC)/2
out=p.sort_values('S',ascending=False)
print(out.head(top).round(3))
if len(sys.argv)>4:
    o=out.copy(); o.columns=[c[0] if c[1]=='' else '_'.join(c) for c in o.columns]; print(o.reset_index().groupby('model')[['ARI','ACC','S']].max().sort_values('S',ascending=False).round(3))
