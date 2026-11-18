import pandas as pd,sys
d=pd.read_json(sys.argv[1],lines=True)
print(d.sort_values('ari',ascending=False).head(8)[['red','clu','ari','nmi','acc','noise','k','t_reduce','t_cluster']].round(3))
print(d[d.clu.str.startswith(('kmeans','agglo'))][['red','clu','ari','acc','k']].round(3))
