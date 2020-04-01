import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

Data = {'x': [13,22,30,20,31,35,25,47,46,54,53,56,58,55,58,63,67,73,76],
        'y': [41,38,32,54,58,49,59,62,65,73,70,65,61,27,17,9,21,9,27]}

path = pd.DataFrame(Data,columns=['x','y'])
dendrogram = sch.dendrogram(sch.linkage(path, method='ward'))
plt.show()

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(path)

plt.scatter(path['x'], path['y'], c=hc.labels_, s= 100)
plt.show()