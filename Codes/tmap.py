# Import the class
import kmapper as km
import pandas
import sklearn
import numpy


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
data=data.loc[labels.index,:].values

tooltip_s = numpy.array(
    labels.index
)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection="l2norm") # X-Y axis

#Can define custom lenses here


# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data, cover=km.Cover(n_cubes=10,perc_overlap=0.1),
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
mapper.visualize(graph, path_html="./../Results/data_keplermapper_output.html",   color_values=labels,color_function_name=["Bronchieactasis or BCOS or Copd"],title="",custom_tooltips=tooltip_s,colorscale=colorscale_default,
custom_meta={"Metadata":"you can add"},X=data,X_names=allergens,lens=projected_data,lens_names=["L2 norm"])

'''
mapper.visualize(graph, path_html="./../Results/data_keplermapper_output.html",   color_values=labels.values,color_function_name=[i for i in labels.columns],title="",custom_tooltips=tooltip_s,colorscale=colorscale_default,
custom_meta={"Metadata":"you can add"},X=data,X_names=allergens,lens=projected_data,lens_names=["L2 norm"])
'''
