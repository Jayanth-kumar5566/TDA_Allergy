# Import the class
import kmapper as km
import pandas
import sklearn
import numpy
import matplotlib.pyplot as plt

#========== Define Data and Labels here==========
data=pandas.read_csv("./../Results/feat_sel_data.csv",index_col=0)

allergens=data.columns.values

labels=data[["labels"]]

#O - bronchieactasis, 1- BCOS and 2 - COPD
data=data.iloc[:,:-1].values

tooltip_s = numpy.array(
    labels.index
)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
#projected_data = mapper.fit_transform(data, projection="l2norm") # X-Y axis

#Can define custom lenses here
projected_data = numpy.linalg.norm(data,ord=2,axis=1,keepdims=True)
plt.hist(projected_data,bins=100)
plt.savefig("./../Results/lens_hist.png",dpi=300)



# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data, cover=km.Cover(n_cubes=9,perc_overlap=0.08),
    clusterer=sklearn.cluster.KMeans(n_clusters=2, random_state=5454564))

#Color_Scale
colorscale_default = [
    [0.0, "rgb(68, 1, 84)"],  # Viridis
    [0.1, "rgb(72, 35, 116)"],
    [0.2, "rgb(64, 67, 135)"],
    [0.3, "rgb(52, 94, 141)"],
    [0.4, "rgb(41, 120, 142)"],
    [0.5, "rgb(32, 144, 140)"],
    [0.6, "rgb(34, 167, 132)"],
    [0.7, "rgb(68, 190, 112)"],
    [0.8, "rgb(121, 209, 81)"],
    [0.9, "rgb(189, 222, 38)"],
    [1.0, "rgb(253, 231, 36)"],
]

#Look up the below link - functions already written to convert color scales to the above format
#https://kepler-mapper.scikit-tda.org/en/latest/_modules/kmapper/visuals.html


# Visualize it
'''
mapper.visualize(graph, path_html="./../Results/feat_sel_data_keplermapper_output.html",   color_values=labels,color_function_name=["Bronchieactasis or BCOS or Copd"],title="",custom_tooltips=tooltip_s,colorscale=colorscale_default,
custom_meta={"Metadata":"you can add"},X=data,X_names=allergens,lens=projected_data,lens_names=["L2 norm"])

'''
mapper.visualize(graph, path_html="./../Results/feat_sel_data_keplermapper_output.html",   color_values=labels.values,color_function_name=[i for i in labels.columns],title="",custom_tooltips=tooltip_s,colorscale=colorscale_default,
custom_meta={"Metadata":"you can add"},X=data,X_names=allergens,lens=projected_data,lens_names=["L2 norm"])
''

#=====Parse graph object to original data to map back clusters=====
cluster=pandas.Series(graph['nodes'])

dict_cluster={}
for i in range(len(labels.index)):
    dict_cluster[i]=labels.index[i]

def replace_dict(x,dict_cluster=dict_cluster):
    y=[dict_cluster[i] for i in x]
    return(y)

cluster=cluster.apply(replace_dict)

#Convert to a networkx graph as 
import networkx as nx
nx_graph=km.adapter.to_nx(graph)
#create adjacency matrix and send to cytoscape
A=nx.to_numpy_matrix(nx_graph)
A=pandas.DataFrame(A,columns=nx_graph.nodes,index=nx_graph.nodes)
#A is the adjacency matrix
A.to_csv("./../Results/Adjacency_matrix.csv")

#==Now creating the feature matrix
data=pandas.read_csv("./../Results/feat_sel_data.csv",index_col=0)
data=data.iloc[:,:-1]

simp_data=dict()
for i in cluster.index:
    simp_data[i]=data.loc[cluster[i],:].median(axis=0)


simp_data=pandas.DataFrame(simp_data).transpose()
#simp_data - is the median metadata for the simplices
simp_data.to_csv("./../Results/Adjacency_metadata.csv")
