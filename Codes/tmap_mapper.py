# Import the class
import pandas
import numpy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from tmap.tda import mapper
from tmap.tda.Filter import _filter_dict
from tmap.tda.cover import Cover
from tmap.tda.metric import Metric
from tmap.tda.utils import optimize_dbscan_eps
import matplotlib.pyplot as plt

#========== Define Data and Labels here==========
b_data=pandas.read_csv("./../Results/bronchieactasis_data.csv",index_col=0)

c_data=pandas.read_csv("./../Results/COPD.csv",index_col=0)

#=======Data creation merging=======
data=pandas.concat([b_data,c_data])

allergens=data.columns.values

labels_b=pandas.read_csv("./../../Bronchiectasis_Clinical_Metadata_V3.4.csv",index_col=0)
labels_c=pandas.read_csv("./../../COPD_Clinical_Metadata_V3.3_tpyxlsx.csv",index_col=0)

labels_b=labels_b[["BCOS"]]
labels_c=labels_c[["COPD_Comorbidities_CoExisting_Bronchiectasis"]]

#Create new column names and concat
labels_b.columns=["y"]
labels_b=labels_b.replace({"No":0,"Yes":1})
labels_c.columns=["y"]
labels_c=labels_c.replace({"No":2,"Yes":1})
labels=pandas.concat([labels_b,labels_c])

#O - bronchieactasis, 1- BCOS and 2 - COPD
data_meta=data.loc[labels.index,:]
data=data_meta.values

#tooltip_s = numpy.array(
#    labels.index
#)

# Initialize
tm= mapper.Mapper(verbose=1)

# Custom Lens
projected_data = numpy.linalg.norm(data,ord=2,axis=1,keepdims=True)
plt.hist(projected_data,bins=100)
plt.savefig("./../Results/lens_hist.png",dpi=300)

# Covering, Clustering and Mapping
eps = optimize_dbscan_eps(data, threshold=95)
clusterer = DBSCAN(eps=eps, min_samples=3)
cover = Cover(projected_data=MinMaxScaler().fit_transform(projected_data), resolution=10, overlap=0.1)
#resolution - number of hypercubes/nodes
graph = tm.map(data=data, cover=cover, clusterer=clusterer)

#Plotting
from tmap.tda.plot import show, Color,tm_plot,vis_progressX
color = Color(target=labels, dtype="categorical")
#tm_plot(graph, graph_data=labels.values ,color=color, mode='file',filename='vis_2.html')

#SAFE with metadata
from tmap.netx.SAFE import SAFE_batch, get_SAFE_summary
n_iter = 1000
enriched_scores = SAFE_batch(graph,
                     metadata=data_meta,
                     n_iter=n_iter,
                     _mode = 'enrich')


safe_summary = get_SAFE_summary(graph=graph,
                            metadata=data_meta,
                            safe_scores=enriched_scores,
                            n_iter=n_iter,
                            p_value=0.01)

#Significant nodes only - calculation
n_iter = 1000
safe_scores = SAFE_batch(graph, metadata=data_meta, n_iter=n_iter)
from tmap.netx.SAFE import get_significant_nodes
min_p_value = 1.0 / (n_iter + 1.0)
p_value = 0.05
SAFE_pvalue = numpy.log10(p_value) / numpy.log10(min_p_value)
enriched_centroides, enriched_nodes = get_significant_nodes(graph,safe_scores,SAFE_pvalue=SAFE_pvalue,r_neighbor=True)
