import numpy as np
import networkx as nx
from itertools import product
import time
from Clustering_spectral import cluster_kmeans, cluster_hierarchical_eucl, cluster_spectral, main_func, main_func2
from Clustering_random_graphs import save_results
from Clustering_state_of_art  import compute_graph2vec_results, nx_to_grakel, run_graph_kernels, run_graph_kernels2


# THIS FILE CONTAINS CLUSTERING OF OTHER BIOLOGICAL GRAPH COLLECTIONS FROM TU DATASET (E.G. MUTAG)


# FUNCTION TO LOAD TU DATASET (IF NO NODE LABELS ARE GIVEN)
# def load_tu_dataset(path, name):
#     # Edges
#     edges = []
#     with open(f"{path}/{name}_A.txt") as f:
#         for line in f:
#             u, v = map(int, line.strip().split(","))
#             edges.append((u, v))

#     # Graph indicators
#     graph_indicator = {}
#     with open(f"{path}/{name}_graph_indicator.txt") as f:
#         for i, line in enumerate(f, start=1):
#             graph_indicator[i] = int(line.strip())

#     # Graph labels
#     labels = []
#     with open(f"{path}/{name}_graph_labels.txt") as f:
#         for line in f:
#             labels.append(int(line.strip()))

#     # Build graphs
#     graphs = {}
#     for node, g_id in graph_indicator.items():
#         if g_id not in graphs:
#             graphs[g_id] = nx.Graph()
#         graphs[g_id].add_node(node)

#     for u, v in edges:
#         g_id = graph_indicator[u]
#         graphs[g_id].add_edge(u, v)

#     # Convert to list of graphs
#     graphs = [nx.convert_node_labels_to_integers(G) for G in graphs.values()]

#     return graphs, labels


# FUNCTION TO LOAD TU DATASET (IF NODE LABELS ARE GIVEN)
def load_tu_dataset(path, name):
    # Edges
    edges = []
    with open(f"{path}/{name}_A.txt") as f:
        for line in f:
            u, v = map(int, line.strip().split(","))
            edges.append((u, v))

    # Graph indicators
    graph_indicator = {}
    with open(f"{path}/{name}_graph_indicator.txt") as f:
        for i, line in enumerate(f, start=1):
            graph_indicator[i] = int(line.strip())

    # Graph labels
    graph_labels = []
    with open(f"{path}/{name}_graph_labels.txt") as f:
        for line in f:
            graph_labels.append(int(line.strip()))

    # Node labels
    node_labels = {}
    with open(f"{path}/{name}_node_labels.txt") as f:
        for i, line in enumerate(f, start=1):
            node_labels[i] = str(line.strip())  # Must be string

    # Build graphs
    graphs = {}
    for node, g_id in graph_indicator.items():
        if g_id not in graphs:
            graphs[g_id] = nx.Graph()
        graphs[g_id].add_node(node, label=node_labels[node])

    for u, v in edges:
        g_id = graph_indicator[u]
        graphs[g_id].add_edge(u, v)

    # Convert to list of graphs
    graphs = [
        nx.convert_node_labels_to_integers(G)
        for G in graphs.values()
    ]

    return graphs, graph_labels


# READ DATASET
#graphs, labels = load_tu_dataset("C:/Users/edina/Downloads/ENZYMES/ENZYMES", "ENZYMES")
#graphs, labels = load_tu_dataset("C:/Users/edina/Downloads/AIDS/AIDS", "AIDS")
#graphs, labels = load_tu_dataset("C:/Users/edina/Downloads/BZR/BZR", "BZR")
#graphs, labels = load_tu_dataset("C:/Users/edina/Downloads/COX2/COX2", "COX2")
#graphs, labels = load_tu_dataset("C:/Users/edina/Downloads/DHFR/DHFR", "DHFR")
#graphs, labels = load_tu_dataset("C:/Users/edina/Downloads/SYNTHETIC/SYNTHETIC", "SYNTHETIC")
#graphs, labels = load_tu_dataset("C:/Users/edina/Downloads/NCI1/NCI1", "NCI1")
graphs, labels = load_tu_dataset("C:/Users/edina/Downloads/MUTAG/MUTAG", "MUTAG")

number_of_clusters = 2

print(len(graphs))
print(len(labels))

# Filter to keep only graphs with at least k nodes
filtered = [(G, y) for G, y in zip(graphs, labels) if G.number_of_nodes() >= 10]
graphs, labels = zip(*filtered)
print(len(graphs))



###########################################

# CHECK SOME STATS ABOUT GRAPHS 
import numpy as np
import networkx as nx
from collections import defaultdict

def check_graph_statistics(graphs, labels):
    stats = defaultdict(lambda: {
        "num_nodes": [],
        "num_edges": [],
        "avg_degree": [],
        "density": []
    })

    for G, label in zip(graphs, labels):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        avg_deg = np.mean([d for _, d in G.degree()])
        dens = nx.density(G)

        stats[label]["num_nodes"].append(n)
        stats[label]["num_edges"].append(m)
        stats[label]["avg_degree"].append(avg_deg)
        stats[label]["density"].append(dens)

    # Print results
    for label in sorted(stats.keys()):
        print(f"\nClass {label}:")
        print(f"Avg #nodes: {np.mean(stats[label]['num_nodes']):.3f}")
        print(f"Avg #edges: {np.mean(stats[label]['num_edges']):.3f}")
        print(f"Avg degree: {np.mean(stats[label]['avg_degree']):.3f}")
        print(f"Avg density: {np.mean(stats[label]['density']):.5f}")

        print(f"Std #nodes: {np.std(stats[label]['num_nodes']):.3f}")
        print(f"Std #edges: {np.std(stats[label]['num_edges']):.3f}")
        print(f"Std degree: {np.std(stats[label]['avg_degree']):.3f}")
        print(f"Std density: {np.std(stats[label]['density']):.5f}")

check_graph_statistics(graphs, labels)


node_counts = [G.number_of_nodes() for G in graphs]
edge_counts = [G.number_of_edges() for G in graphs]

min_nodes = min(node_counts)
max_nodes = max(node_counts)
avg_nodes = sum(node_counts) / len(node_counts)
avg_edges = sum(edge_counts) / len(edge_counts)

print("Min nodes:", min_nodes)
print("Max nodes:", max_nodes)
print("Average nodes:", avg_nodes)
print("Average edges:", avg_edges)


##########################################



#################################################################
########################   EIGENVALUES   ########################
#################################################################

# Define parameter grid
param_grid = {
    "k": [4],
    "which_L": ["normalized"],
    "which_eigvals": ["SM",  "LM", "BE"],
    "embedding_method": ["eigenvalues"], 
    "clustering_method": [cluster_kmeans, cluster_hierarchical_eucl, cluster_spectral],
    "number_of_clusters": [number_of_clusters]
}


# Do experiment 
n_runs = 30
results = []

for values in product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), values))

    
    ari_scores = []
    nmi_scores = []
    runtimes = []
    
    for _ in range(n_runs):
        start_time = time.time()
        result = main_func(
            graphs,
            labels,
            k=params["k"],
            which_L=params["which_L"],
            which_eigvals=params["which_eigvals"],
            embedding_method=params["embedding_method"],
            bins=params["k"],
            clustering_method=params["clustering_method"],
            number_of_clusters=params["number_of_clusters"]
        )
        end_time = time.time()
        
        ari, nmi = result
        
        ari_scores.append(ari)
        nmi_scores.append(nmi)
        
        runtimes.append(end_time - start_time)

    avg_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    
    avg_nmi = np.mean(nmi_scores)
    std_nmi = np.std(nmi_scores)
    
    avg_time = np.mean(runtimes)
    std_time = np.std(runtimes)
    
    print(f"ARI: {avg_ari:.4f}, {std_ari:.4f}")
    print(f"NMI: {avg_nmi:.4f}, {std_nmi:.4f}")
    print(f"Time: {avg_time:.4f}s, {std_time:.4f}s")
    
    results.append({
        "params": params,
        "avg_ari": avg_ari,
        "std_ari": std_ari,
        "avg_nmi": avg_nmi,
        "std_nmi": std_nmi,
        "avg_time": avg_time,
        "std_time": std_time
    })


print(results)
save_results(results, filename="Clustering_results/results_mutag_k4_eigvals.json")


print(main_func(graphs, labels, k=4, which_L = "normalized", which_eigvals = "LM", embedding_method = "eigenvalues", 
              bins = 30, clustering_method = cluster_kmeans, number_of_clusters = number_of_clusters))


#################################################################
#####################   EIGENVALUE COUNTS   #####################
#################################################################

# Define param grid
param_grid = {
    "k": [9],
    "which_L": ["normalized"],
    "which_eigvals": ["SM", "LM", "BE"],
    "embedding_method": ["eigenvalue counts"],
    "bins": [9],
    "clustering_method": [cluster_kmeans, cluster_hierarchical_eucl, cluster_spectral],
    "number_of_clusters": [number_of_clusters]
}


# Do experiment 
n_runs = 30
results = []

for values in product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), values))
    
    
    ari_scores = []
    nmi_scores = []
    runtimes = []
    
    for _ in range(n_runs):
        start_time = time.time()
        result = main_func(
            graphs,
            labels,
            k=params["k"],
            which_L=params["which_L"],
            which_eigvals=params["which_eigvals"],
            embedding_method=params["embedding_method"],
            bins=params["bins"], 
            clustering_method=params["clustering_method"],
            number_of_clusters=params["number_of_clusters"]
        )
        end_time = time.time()
        
        ari, nmi = result
        
        ari_scores.append(ari)
        nmi_scores.append(nmi)
        
        runtimes.append(end_time - start_time)

    avg_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    
    avg_nmi = np.mean(nmi_scores)
    std_nmi = np.std(nmi_scores)
    
    avg_time = np.mean(runtimes)
    std_time = np.std(runtimes)
    
    print(f"ARI: {avg_ari:.4f}, {std_ari:.4f}")
    print(f"NMI: {avg_nmi:.4f}, {std_nmi:.4f}")
    print(f"Time: {avg_time:.4f}s, {std_time:.4f}s")
    
    results.append({
        "params": params,
        "avg_ari": avg_ari,
        "std_ari": std_ari,
        "avg_nmi": avg_nmi,
        "std_nmi": std_nmi,
        "avg_time": avg_time,
        "std_time": std_time
    })


print(results)
save_results(results, filename="Clustering_results/results_mutag_k9_eig_counts.json")

print(main_func(graphs, labels, k=9, which_L = "normalized", which_eigvals = "LM", embedding_method = "eigenvalue counts", 
              bins = 10, clustering_method = cluster_kmeans, number_of_clusters = number_of_clusters))


#################################################################
#########################   GRAPH2VEC   #########################
#################################################################

# One test run
result_graph2vec = compute_graph2vec_results(graphs, labels, number_of_clusters = number_of_clusters, dim = 64)
print(result_graph2vec)


n_runs = 30
    
ari_scores_kmeans = []
nmi_scores_kmeans = []
ari_scores_hierarchical = []
nmi_scores_hierarchical = []
ari_scores_spectral = []
nmi_scores_spectral = []
runtimes = []

for _ in range(n_runs):
    start_time = time.time()
    ari_kmeans, nmi_kmeans, ari_hierarchical, nmi_hierarchical, ari_spectral, nmi_spectral = compute_graph2vec_results(graphs, labels, number_of_clusters = number_of_clusters, dim = 30)
    end_time = time.time()

    ari_scores_kmeans.append(ari_kmeans)
    nmi_scores_kmeans.append(nmi_kmeans)
    
    ari_scores_hierarchical.append(ari_hierarchical)
    nmi_scores_hierarchical.append(nmi_hierarchical)

    ari_scores_spectral.append(ari_spectral)
    nmi_scores_spectral.append(nmi_spectral)
    
    runtimes.append((end_time - start_time)/3)


avg_ari_kmeans = np.mean(ari_scores_kmeans)
std_ari_kmeans = np.std(ari_scores_kmeans)
avg_nmi_kmeans = np.mean(nmi_scores_kmeans)
std_nmi_kmeans = np.std(nmi_scores_kmeans)

avg_ari_hierarchical = np.mean(ari_scores_hierarchical)
std_ari_hierarchical = np.std(ari_scores_hierarchical)
avg_nmi_hierarchical = np.mean(nmi_scores_hierarchical)
std_nmi_hierarchical = np.std(nmi_scores_hierarchical)

avg_ari_spectral = np.mean(ari_scores_spectral)
std_ari_spectral = np.std(ari_scores_spectral)
avg_nmi_spectral = np.mean(nmi_scores_spectral)
std_nmi_spectral = np.std(nmi_scores_spectral)

avg_time = np.mean(runtimes)
std_time = np.std(runtimes)

print(f"ARI KMEANS: {avg_ari_kmeans:.4f}, {std_ari_kmeans:.4f}")
print(f"NMI KMEANS: {avg_nmi_kmeans:.4f}, {std_nmi_kmeans:.4f}")
print(f"ARI HIERARCHICAL: {avg_ari_hierarchical:.4f}, {std_ari_hierarchical:.4f}")
print(f"NMI HIERARCHICAL: {avg_nmi_hierarchical:.4f}, {std_nmi_hierarchical:.4f}")
print(f"ARI SPECTRAL: {avg_ari_spectral:.4f}, {std_ari_spectral:.4f}")
print(f"NMI SPECTRAL: {avg_nmi_spectral:.4f}, {std_nmi_spectral:.4f}")
print(f"Time: {avg_time:.4f}s, {std_time:.4f}s")



#################################################################
##########################   KERNELS   ##########################
#################################################################

# Convert networkx graphs to grakel graphs
gk_graphs = [nx_to_grakel(G) for G in graphs]

# One test run
result_kernels = run_graph_kernels(gk_graphs, labels, number_of_clusters=2, kernel = "wl")
print(result_kernels)



n_runs = 30



# WL kernel clustering  

ari_scores_hierarchical = []
nmi_scores_hierarchical = []
ari_scores_spectral = []
nmi_scores_spectral = []
runtimes = []


for _ in range(n_runs):
    start_time = time.time()
    ari_hierarchical, nmi_hierarchical, ari_spectral, nmi_spectral = run_graph_kernels(gk_graphs, labels, number_of_clusters=number_of_clusters, kernel = "wl")
    end_time = time.time()
    
    
    ari_scores_hierarchical.append(ari_hierarchical)
    nmi_scores_hierarchical.append(nmi_hierarchical)

    ari_scores_spectral.append(ari_spectral)
    nmi_scores_spectral.append(nmi_spectral)
    
    runtimes.append((end_time - start_time)/2)


avg_ari_hierarchical = np.mean(ari_scores_hierarchical)
std_ari_hierarchical = np.std(ari_scores_hierarchical)
avg_nmi_hierarchical = np.mean(nmi_scores_hierarchical)
std_nmi_hierarchical = np.std(nmi_scores_hierarchical)

avg_ari_spectral = np.mean(ari_scores_spectral)
std_ari_spectral = np.std(ari_scores_spectral)
avg_nmi_spectral = np.mean(nmi_scores_spectral)
std_nmi_spectral = np.std(nmi_scores_spectral)

avg_time = np.mean(runtimes)
std_time = np.std(runtimes)

print(f"ARI: {avg_ari_hierarchical:.4f}, {std_ari_hierarchical:.4f}")
print(f"NMI: {avg_nmi_hierarchical:.4f}, {std_nmi_hierarchical:.4f}")
print(f"ARI: {avg_ari_spectral:.4f}, {std_ari_spectral:.4f}")
print(f"NMI: {avg_nmi_spectral:.4f}, {std_nmi_spectral:.4f}")
print(f"Time: {avg_time:.4f}s, {std_time:.4f}s")



# SP kernel clustering

ari_scores_hierarchical = []
nmi_scores_hierarchical = []
ari_scores_spectral = []
nmi_scores_spectral = []
runtimes = []


for _ in range(n_runs):
    start_time = time.time()
    ari_hierarchical, nmi_hierarchical, ari_spectral, nmi_spectral = run_graph_kernels(gk_graphs, labels, number_of_clusters=number_of_clusters, kernel = "sp")
    end_time = time.time()
    
    ari_scores_hierarchical.append(ari_hierarchical)
    nmi_scores_hierarchical.append(nmi_hierarchical)

    ari_scores_spectral.append(ari_spectral)
    nmi_scores_spectral.append(nmi_spectral)
    
    runtimes.append((end_time - start_time)/2)


avg_ari_hierarchical = np.mean(ari_scores_hierarchical)
std_ari_hierarchical = np.std(ari_scores_hierarchical)
avg_nmi_hierarchical = np.mean(nmi_scores_hierarchical)
std_nmi_hierarchical = np.std(nmi_scores_hierarchical)

avg_ari_spectral = np.mean(ari_scores_spectral)
std_ari_spectral = np.std(ari_scores_spectral)
avg_nmi_spectral = np.mean(nmi_scores_spectral)
std_nmi_spectral = np.std(nmi_scores_spectral)

avg_time = np.mean(runtimes)
std_time = np.std(runtimes)

print(f"ARI: {avg_ari_hierarchical:.4f}, {std_ari_hierarchical:.4f}")
print(f"NMI: {avg_nmi_hierarchical:.4f}, {std_nmi_hierarchical:.4f}")
print(f"ARI: {avg_ari_spectral:.4f}, {std_ari_spectral:.4f}")
print(f"NMI: {avg_nmi_spectral:.4f}, {std_nmi_spectral:.4f}")
print(f"Time: {avg_time:.4f}s, {std_time:.4f}s")



# GL kernel clustering

ari_scores_hierarchical = []
nmi_scores_hierarchical = []
ari_scores_spectral = []
nmi_scores_spectral = []
runtimes = []


for _ in range(n_runs):
    start_time = time.time()
    ari_hierarchical, nmi_hierarchical, ari_spectral, nmi_spectral = run_graph_kernels(gk_graphs, labels, number_of_clusters=number_of_clusters, kernel = "gl")
    end_time = time.time()
    
    
    ari_scores_hierarchical.append(ari_hierarchical)
    nmi_scores_hierarchical.append(nmi_hierarchical)

    ari_scores_spectral.append(ari_spectral)
    nmi_scores_spectral.append(nmi_spectral)
    
    runtimes.append((end_time - start_time)/2)


avg_ari_hierarchical = np.mean(ari_scores_hierarchical)
std_ari_hierarchical = np.std(ari_scores_hierarchical)
avg_nmi_hierarchical = np.mean(nmi_scores_hierarchical)
std_nmi_hierarchical = np.std(nmi_scores_hierarchical)

avg_ari_spectral = np.mean(ari_scores_spectral)
std_ari_spectral = np.std(ari_scores_spectral)
avg_nmi_spectral = np.mean(nmi_scores_spectral)
std_nmi_spectral = np.std(nmi_scores_spectral)

avg_time = np.mean(runtimes)
std_time = np.std(runtimes)

print(f"ARI: {avg_ari_hierarchical:.4f}, {std_ari_hierarchical:.4f}")
print(f"NMI: {avg_nmi_hierarchical:.4f}, {std_nmi_hierarchical:.4f}")
print(f"ARI: {avg_ari_spectral:.4f}, {std_ari_spectral:.4f}")
print(f"NMI: {avg_nmi_spectral:.4f}, {std_nmi_spectral:.4f}")
print(f"Time: {avg_time:.4f}s, {std_time:.4f}s")










#############################################
################## SVC TEST #################
#############################################

# to check if there is information in the embeddings and kernel matrices

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from grakel.kernels import WeisfeilerLehman
from grakel.kernels import ShortestPath
import numpy as np



# WL SVC
S = WeisfeilerLehman(n_iter=2, base_graph_kernel=None).fit_transform(gk_graphs)
labels = np.array(labels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, test_idx in cv.split(S, labels):
    K_train = S[np.ix_(train_idx, train_idx)]
    K_test  = S[np.ix_(test_idx, train_idx)]
    
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, labels[train_idx])
    scores.append(clf.score(K_test, labels[test_idx]))

scores = np.array(scores)
print("Accuracy of SVC with WL kernel", scores.mean())



# SP SVC
sp_kernel = ShortestPath()
K_sp = sp_kernel.fit_transform(gk_graphs) 

labels = np.array(labels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_sp = []

for train_idx, test_idx in cv.split(K_sp, labels):
    K_train = K_sp[np.ix_(train_idx, train_idx)]
    K_test  = K_sp[np.ix_(test_idx, train_idx)]
    
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, labels[train_idx])
    scores_sp.append(clf.score(K_test, labels[test_idx]))

scores_sp = np.array(scores_sp)
print("Accuracy of SVC with SP kernel", scores_sp.mean())



# GK SVC
from grakel.kernels import GraphletSampling
gl_kernel = GraphletSampling(k=4) 
K_gl = gl_kernel.fit_transform(gk_graphs)

scores_gl = []

for train_idx, test_idx in cv.split(K_gl, labels):
    K_train = K_gl[np.ix_(train_idx, train_idx)]
    K_test  = K_gl[np.ix_(test_idx, train_idx)]
    
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, labels[train_idx])
    scores_gl.append(clf.score(K_test, labels[test_idx]))

scores_gl = np.array(scores_gl)
print("Accuracy of SVC with GL kernel", scores_gl.mean())



# SPECTRAL EMBEDDING SVC
from sklearn.model_selection import cross_val_score
from karateclub import Graph2Vec
from Clustering_spectral import embed_graphs

clf = SVC(kernel='rbf')  # 'linear', 'rbf'

# Spectral embedding
X = np.array(embed_graphs(graphs, 4, which_L = "normalized", which_eigvals = "SM", embedding_method = "eigenvalues", bins = 10))
scores = cross_val_score(clf, X, labels, cv=5) 

print("Accuracy of SVC with spectral embedding", scores.mean())