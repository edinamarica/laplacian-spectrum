import json
import matplotlib.pyplot as plt
import numpy as np

# THIS FILE CONTAINS THE CODE FOR PLOTTING THE CLUSTERING RESULTS


# PLOT THE SPECTRAL EMBEDDING AND BINNED SPECTRAL EMBEDDING RESULTS 

with open("Clustering_results/results_mutag_k9_eig_counts.json", "r") as f:
    results = json.load(f)

# Bar plot with errors
def plot_metrics(results, k, embedding_method, metric="ari"):
    filtered = [r for r in results if r['params']['k']==k and r['params']['embedding_method']==embedding_method]
    desired_order = ['SM', 'LM', 'BE']
    x_labels = list(set(r['params']['which_eigvals'] for r in filtered))
    x_labels.sort(key=lambda x: desired_order.index(x))
    
    clustering_methods = ['cluster_kmeans', 'cluster_hierarchical_eucl', 'cluster_spectral']
    clustering_labels = ['K-Means', 'Hierarchical', 'Spectral'] 
    
    colors = ["#e6ab02", "#d95f02", "#a6761d"]

    x = np.arange(len(x_labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    for i, (method, color) in enumerate(zip(clustering_methods, colors)):
        means = []
        stds = []
        for we in x_labels:
            # Find the corresponding result
            r = [r for r in filtered if r['params']['which_eigvals']==we and r['params']['clustering_method']==method][0]
            means.append(r[f'avg_{metric}'])
            stds.append(r[f'std_{metric}'])
        ax.bar(x + i*width, means, width, yerr=stds, label=method, capsize=5, color=color)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(x_labels, fontsize = 22)
    ax.set_ylabel(metric.upper(), fontsize = 24)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_xlabel("Which eigenvalues", fontsize = 24)
    ax.set_title(f"{metric.upper()} for {k} eigenvalues, embedding: {embedding_method}", fontsize = 26)
    ax.set_title(f"{metric.upper()} for {k} eigenvalues, embedding: binned spectral", fontsize = 24)
    ax.legend(labels = clustering_labels, fontsize = 18, loc = "upper right")
    plt.tight_layout()
    plt.ylim(0, 1.2) 
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    safe_embedding = embedding_method.replace(" ", "_")
    filename_base = f"{metric.upper()}_k{k}_{safe_embedding}"
    plt.savefig(f"Clustering_figures/{filename_base}_mutag.png", dpi=300)
    #plt.savefig(f"Clustering_figures/{filename_base}_mutag.eps", format='eps')
    plt.show()


# ARI plots
plot_metrics(results, k=4, embedding_method="eigenvalues", metric="ari")
plot_metrics(results, k=9, embedding_method="eigenvalue counts", metric="ari")

# NMI plots
plot_metrics(results, k=4, embedding_method="eigenvalues", metric="nmi")
plot_metrics(results, k=9, embedding_method="eigenvalue counts", metric="nmi")





# PLOT THE COMPARISON WITH OTHER METHODS

# MUTAG 
scores=[[0.42, 0.29],
       [0.48, 0.29],
       [0.022, 0.017],
       [0.145, 0.087],
       [0.22,  0.14],
       [0.002,  0.007]]


methods = ["Spectral \n embedding", "Binned spectral \n embedding", "Graph2vec \n embedding", "WL kernel", "SP kernel", "Graphlet kernel"]


def plot_scores_barplots(scores, methods, metrics=["ARI", "NMI", "Time"], units=["", "", "s"]):
    scores = np.array(scores)
    x = np.arange(len(methods))

    metric_cols = {
        "ARI": (0),
        "NMI": (1)
    }

    for metric, unit in zip(metrics, units):
        avg_col = metric_cols[metric]
        avg = scores[:, avg_col]

        plt.figure(figsize=(10,8))
        plt.bar(x, avg, capsize=5)
        plt.xticks(x, methods, rotation=45, ha='right', fontsize = 20)
        plt.tick_params(axis='y', labelsize=20)
        plt.ylabel(f"{metric} {unit}".strip(), fontsize = 22)
        plt.ylim(0,1)
        plt.title(f"Comparison of {metric} scores", fontsize = 24)
        plt.tight_layout()
        plt.savefig(f"Clustering_figures/{metric}_MUTAG_comp.png", dpi=300)
        plt.savefig(f"Clustering_figures/{metric}_MUTAG_comp.eps", format='eps')
        plt.show()


plot_scores_barplots(scores, methods, metrics=["ARI", "NMI"], units=["", ""])