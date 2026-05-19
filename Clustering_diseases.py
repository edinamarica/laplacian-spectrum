import numpy as np
import networkx as nx
from itertools import product
from Clustering_spectral import cluster_kmeans, cluster_hierarchical_eucl, cluster_spectral, main_func
from Clustering_state_of_art  import compute_graph2vec_results
from Clustering_state_of_art  import nx_to_grakel, run_graph_kernels
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import pickle
import json

# THIS FILE CONTAINS THE CLUSTERING OF THE CANCER DISEASES


# IMPORT FILE WITH CANCER DISEASES ONLY - THIS WILL BE FILTERED OUT TO ONLY KEEP DISEASES WITH LCC>=50
with open("data/cancer_only.pkl", "rb") as f:
    cancer_only = pickle.load(f)

graphs = []
labels = []

for label, G in cancer_only.items():
    labels.append(label)
    graphs.append(G)

print(len(graphs))



# IMPORT SIMILARITY MATRIX AND COMPUTE CLUSTERING BASED ON IT FOR GROUND TRUTH CLUSTERING
data_sim = np.load("data/lin_similarity_with_labels.npz", allow_pickle=True)
sim_mx  = data_sim["S_l"] 
dis = data_sim["doids"]
dist_mx = 1 - sim_mx

model = SpectralClustering(
    n_clusters=12,
    affinity='precomputed'
)
labels = model.fit_predict(sim_mx)
print(len(labels))
#print(silhouette_score(dist_mx, labels, metric="precomputed"))



# IMPORT DOID TO NAME DICTIONARY TO CONVERT DOIDS TO DISEASE NAMES
with open("data/doid_to_name.json", "r") as f:   
    doid_to_name = json.load(f)

dis_to_name = [doid_to_name[d] for d in dis]


# FILTER TO ONLY KEEP DISEASES THAT ARE IN THE SIMILARITY MATRIX (i.e. lcc>=50)
filtered_graphs = []
filtered_labels = []

for G, label in zip(graphs, labels):
    if label in dis_to_name:
        filtered_graphs.append(G)
        filtered_labels.append(label)

print(len(filtered_graphs))
print(len(labels)) # keep labels corresponding to ground truth



#################################################################
##############  EIGENVALUES / EIGENVALUE COUNTS   ###############
#################################################################

result_sp = main_func(filtered_graphs, labels, k=10, which_L = "normalized", which_eigvals = "LM", embedding_method = "eigenvalues", 
             bins = 5, clustering_method = cluster_kmeans, number_of_clusters = 12)
print(result_sp)


#################################################################
#########################   GRAPH2VEC   #########################
#################################################################

# Relabel nodes of graphs from 0
def canonicalize_graph(g):
    g = g.copy()
    mapping = {node: i for i, node in enumerate(sorted(g.nodes()))}
    return nx.relabel_nodes(g, mapping, copy=True)

graphs_clean = [canonicalize_graph(g) for g in filtered_graphs]


result_graph2vec = compute_graph2vec_results(graphs_clean, labels, number_of_clusters = 12, dim = 256)
print(result_graph2vec)


#################################################################
##########################   KERNELS   ##########################
#################################################################

# Convert networkx graphs to grakel graphs
gk_graphs = [nx_to_grakel(G) for G in graphs_clean]

result_kernels = run_graph_kernels(gk_graphs, labels, number_of_clusters=12, kernel = "sp")
print(result_kernels)

