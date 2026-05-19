import networkx as nx
import numpy as np
from karateclub import Graph2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from gklearn.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel
from sklearn.cluster import SpectralClustering
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, ShortestPath, GraphletSampling, RandomWalk
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from Clustering_spectral import cluster_kmeans, cluster_hierarchical_eucl, cluster_spectral, evaluate, evaluate2


# THIS FILE CONTAINS THE FUNCTIONS REQUIRED FOR CLUSTERING 
# BASED ON GRAPH2VEC EMBEDDING AND KERNELS


#################################################################
#########################   GRAPH2VEC   #########################
#################################################################


def graph2vec_embedding(graphs, dim=64):
    model = Graph2Vec(
        dimensions=dim,
        wl_iterations=2
    )
    model.fit(graphs)
    return model.get_embedding()



clustering_methods = {
    "kmeans": cluster_kmeans,
    "hierarchical_eucl": cluster_hierarchical_eucl,
    "spectral": cluster_spectral
}


# MAIN FUNCTION TO COMPUTE RESULTS
def compute_graph2vec_results(graphs, true_labels, number_of_clusters, dim = 10):

    X = graph2vec_embedding(graphs, dim=dim)
    results = []

    for name, method in clustering_methods.items():
        pred_labels = method(X, number_of_clusters)
        ari, nmi =  evaluate(true_labels, pred_labels)
        results.append(ari)
        results.append(nmi)

    return results



#################################################################
##########################   KERNELS   ##########################
#################################################################


# FUNCTION TO CONVERT NETWORKX TO GRAKEL GRAPH
def nx_to_grakel(G):
    # Get labels
    labels = nx.get_node_attributes(G, 'label')
    if len(labels) == 0:
        # If no labels, use degrees as labels
        labels = {n: str(G.degree(n)) for n in G.nodes()}
    edges = list(G.edges())
    return Graph(edges, node_labels=labels)


# FUNCTION TO DO CLUSTERING
def kernel_clustering(gk_graphs, kernel, n_clusters=2):
    # Similarity matrix
    K = kernel.fit_transform(gk_graphs)  #shape (n_graphs, n_graphs)
    
    # Normalize
    D = np.sqrt(np.outer(np.diag(K), np.diag(K)))
    K_norm = K / (D + 1e-10)
    
    # Convert similairty matrix to distance matrix
    distance_matrix = 1 - K_norm


    labels_hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="complete" #average
    ).fit_predict(distance_matrix)


    labels_spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans"
    ).fit_predict(K_norm)
    
    return labels_hierarchical, labels_spectral


# MAIN FUNCTION FOR KERNEL CLUSTERING
def run_graph_kernels(gk_graphs, true_labels, number_of_clusters=2, kernel = "wl"):
    
    # WL kernel
    if kernel == "wl":
        wl = WeisfeilerLehman(n_iter=2, base_graph_kernel=None)  # default is VertexHistogram
        labels_hierarchical, labels_spectral = kernel_clustering(gk_graphs, wl, number_of_clusters)
        ari_hierarchical, nmi_hierarchical = evaluate(true_labels, labels_hierarchical)
        ari_spectral, nmi_spectral = evaluate(true_labels, labels_spectral)
        return ari_hierarchical, nmi_hierarchical, ari_spectral, nmi_spectral
    
    # SP kernel
    if kernel == "sp":
        sp = ShortestPath()
        labels_hierarchical, labels_spectral = kernel_clustering(gk_graphs, sp, number_of_clusters)
        ari_hierarchical, nmi_hierarchical = evaluate(true_labels, labels_hierarchical)
        ari_spectral, nmi_spectral = evaluate(true_labels, labels_spectral)
        return ari_hierarchical, nmi_hierarchical, ari_spectral, nmi_spectral
    
    # Graphlet kernel
    if kernel == "gl":
        gl = GraphletSampling(k=4) 
        labels_hierarchical, labels_spectral = kernel_clustering(gk_graphs, gl, number_of_clusters)
        ari_hierarchical, nmi_hierarchical = evaluate(true_labels, labels_hierarchical)
        ari_spectral, nmi_spectral = evaluate(true_labels, labels_spectral)
        return ari_hierarchical, nmi_hierarchical, ari_spectral, nmi_spectral
