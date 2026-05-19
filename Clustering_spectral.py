import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, normalize
from itertools import product
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score


# THIS FILE CONTAINS THE FUNCTIONS REQUIRED FOR CLUSTERING
# BASED ON SPECTRAL AND BINNED SPECTRAL EMBEDDINGS




# FUNCTION TO COMPUTE THE (NORMALIZED) LAPLACIAN
def normalized_laplacian(G, which_L = "normalized"):
    if which_L == "normalized":
        return nx.normalized_laplacian_matrix(G).asfptype()
    elif which_L == "classic":
        return nx.laplacian_matrix(G).asfptype()
    else:
        return "please choose between classic or normalized"

# FUNCTION TO COMPUTE FIRST/LAST/FIRST+LAST K EIGENVALUES
def compute_eigenvalues(L, k=None, which_eigvals = "SM"):
    if k is None: 
        selected_eigenvalues = np.linalg.eigvalsh(L.toarray())
    else:
        L.toarray()
        if which_eigvals == 'LM' or which_eigvals == 'BE':
            selected_eigenvalues, _ = eigsh(L, k=k, which=which_eigvals, ncv=30*k)
        elif which_eigvals == 'SM':
            selected_eigenvalues, _ = eigsh(L, k=k, which="LM", ncv=30*k, sigma = 1e-5)
        elif which_eigvals == 'around 1':
            selected_eigenvalues, _ = eigsh(L, k=k, sigma=1, mode = 'buckling', ncv=3*k)
        else:
            return "please choose one from SM, LM, BE, around 1"
    return selected_eigenvalues


# FUNCTION TO COMPUTE HISTOGRAM BIN COUNT VECTOR
def histogram_features(eigvals, bins=10):
    counts, _ = np.histogram(eigvals, bins=bins, range=(0, 2), density=False)
    # Normalization
    total = counts.sum()
    if total > 0:
        counts = counts / total
    else:
        counts = np.zeros_like(counts)
    return counts


# FUNCTION THAT EMBEDS EACH GRAPH TO A VECTOR OF LENGTH K
def embed_graphs(graphs, k, which_L = "normalized", which_eigvals = "SM", embedding_method = "eigenvalues", bins = 10):
    embedded_vectors = []
    k_new = min(k, min(G.number_of_nodes()-1 for G in graphs))
    for G in graphs:
        L = normalized_laplacian(G, which_L = which_L)
        if embedding_method == "eigenvalues":
            embedded_vectors.append(compute_eigenvalues(L, k=k_new, which_eigvals = which_eigvals))
        elif embedding_method == "eigenvalue counts":
            if k < G.number_of_nodes():
                eigvals = compute_eigenvalues(L, k=k, which_eigvals = which_eigvals)
                embedded_vectors.append(histogram_features(eigvals, bins=bins))     
            else: 
                eigvals = compute_eigenvalues(L, k=G.number_of_nodes()-1, which_eigvals = which_eigvals)
                embedded_vectors.append(histogram_features(eigvals, bins=bins))  
        else:
            return "please choose one method from eigenvalues or eigenvalue counts"
    return embedded_vectors





# FUNCTIONS FOR DIFFERENT CLUSTERING METHODS
def cluster_kmeans(X, number_of_clusters):
    X = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=number_of_clusters, n_init=10).fit_predict(X)

def cluster_hierarchical_eucl(X,number_of_clusters):
    X = StandardScaler().fit_transform(X)
    return AgglomerativeClustering(
        n_clusters=number_of_clusters,
        metric="euclidean",
        linkage="average" #complete, average
    ).fit_predict(X)

def cluster_spectral(X,number_of_clusters):
    X = StandardScaler().fit_transform(X)
    return SpectralClustering(
        n_clusters=number_of_clusters,
        affinity="rbf",  
        assign_labels="cluster_qr",  #kmeans, discretize, cluster_qr
        random_state=0
    ).fit_predict(X)


# FUNCTION THAT CLUSTERS THE EMBEDDED GRAPHS
def cluster_vectors(embedded_vectors, clustering_method = cluster_kmeans, number_of_clusters = 2):
    return clustering_method(X = embedded_vectors, number_of_clusters=number_of_clusters)


# EVALUATION OF CLUSTERING
def evaluate(pred_labels, true_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return ari, nmi



def main_func(graphs, true_labels, k, which_L = "normalized", which_eigvals = "SM", embedding_method = "eigenvalues", 
              bins = 10, clustering_method = cluster_kmeans, number_of_clusters = 2):
    embedded_vectors = np.array(embed_graphs(graphs, k=k, which_L = which_L, which_eigvals = which_eigvals, embedding_method = embedding_method, bins = bins))
    labels = cluster_vectors(embedded_vectors, clustering_method = clustering_method, number_of_clusters = number_of_clusters)
    ari, nmi = evaluate(labels, true_labels)
    return ari, nmi