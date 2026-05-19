import numpy as np
import networkx as nx
from itertools import product
import time
from Clustering_spectral import cluster_kmeans, cluster_hierarchical_eucl, cluster_spectral, main_func, compute_eigenvalues, normalized_laplacian
import matplotlib.pyplot as plt
import json
from Clustering_state_of_art  import compute_graph2vec_results, nx_to_grakel, run_graph_kernels


# THIS FILE CONSTAINS CLUSTERING EXPERIMENTS OF RANDOM GRAPHS



# TEST: RANDOM GRAPHS: ER, BA, WS

def generate_graphs(n_graphs=150, n_nodes=30):
    graphs = []
    labels = []

    # Erdos–Renyi
    for _ in range(n_graphs // 3):
        graphs.append(nx.erdos_renyi_graph(n_nodes, 0.05))
        labels.append(0)

    # Barabasi–Albert
    for _ in range(n_graphs // 3):
        graphs.append(nx.barabasi_albert_graph(n_nodes, 5))
        labels.append(1)

    # Watts–Strogatz
    for _ in range(n_graphs // 3):
        graphs.append(nx.watts_strogatz_graph(n_nodes, k=4, p=0.3))
        labels.append(2)

    return graphs, labels

graphs, labels = generate_graphs(n_graphs=150, n_nodes=30)
number_of_clusters = 3




#############################################


# PLOT 1-1 GRAPH OF RANDOM GRAPH COLLECTION
def get_spectrum(G):
    L = nx.normalized_laplacian_matrix(G).asfptype()
    eigenvalues = np.linalg.eigvalsh(L.toarray())
    return np.sort(eigenvalues)

# One graph per type
G_er = graphs[0]
G_ba = graphs[50]
G_ws = graphs[100]

spectra = [
    ("Erdős-Rényi", get_spectrum(G_er), "steelblue"),
    ("Barabási-Albert", get_spectrum(G_ba), "darkorange"),
    ("Watts-Strogatz", get_spectrum(G_ws), "forestgreen")
]

plt.figure(figsize=(6, 5))

for i, (label, spec, color) in enumerate(spectra):
    x = np.full_like(spec, i)  
    plt.scatter(x, spec, color=color, alpha=0.7, s=15) 

plt.xticks([0, 1, 2], ["ER", "BA", "WS"], fontsize = 14)
plt.ylabel("Eigenvalue", fontsize = 14)
plt.title("Spectrum comparison", fontsize = 16)
plt.ylim(0, 2)
plt.show()


# FUNCTION TO SAVE RESULTS OF SPECTRAL-BASED CLUSTERINGS
def save_results(results, filename="results.json"):
    cleaned_results = []

    for r in results:
        r_copy = r.copy()
        r_copy['params'] = r['params'].copy()

        func = r_copy['params']['clustering_method']
        r_copy['params']['clustering_method'] = func.__name__

        cleaned_results.append(r_copy)

    with open(filename, "w") as f:
        json.dump(cleaned_results, f, indent=4)


#############################################







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

    print(ari_scores)
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
save_results(results, filename="Clustering_results/results_synthetic_k4_eigvals.json")



print(main_func(graphs, labels, k=4, which_L = "normalized", which_eigvals = "LM", embedding_method = "eigenvalues", 
              bins = 30, clustering_method = cluster_kmeans, number_of_clusters = number_of_clusters))



#################################################################
#####################   EIGENVALUE COUNTS   #####################
#################################################################

# Define param grid
param_grid = {
    "k": [4],
    "which_L": ["normalized"],
    "which_eigvals": ["SM", "LM", "BE"],
    "embedding_method": ["eigenvalue counts"],
    #"bins": [4],
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

    print(ari_scores)
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
save_results(results, filename="Clustering_results/results_synthetic_k4_eig_counts.json")

print(main_func(graphs, labels, k=4, which_L = "normalized", which_eigvals = "BE", embedding_method = "eigenvalue counts", 
              bins = 4, clustering_method = cluster_kmeans, number_of_clusters = number_of_clusters))




#################################################################
#########################   GRAPH2VEC   #########################
#################################################################

# One test run
result_graph2vec = compute_graph2vec_results(graphs, labels, number_of_clusters = number_of_clusters, dim = 128)
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


gk_graphs = [nx_to_grakel(G) for G in graphs]

# One test run
result_kernels = run_graph_kernels(gk_graphs, labels, number_of_clusters=number_of_clusters, kernel = "wl")
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