import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy.sparse as sp
from scipy.sparse.csgraph import csgraph_from_dense, connected_components
import networkx as nx

def save_tofile(var_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)

def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list
    
def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep

    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep
#def plot_degree(degree_file):
#
#    if os.path.isfile(degree_file):
#        [degree] = load_variable(degree_file)
#
#        plt.hist(collections.Counter(degree), bins='auto')
#        plt.loglog(degree[])
#        plt.title('Histogram of degree')
#        plt.show()

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)['arr_0'].item()
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels
    
if __name__ == "__main__":
    matrix_filename = "cora_ml.npz"
    _A_obs, _X_obs, _z_obs = load_npz(matrix_filename)
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]
    N = _A_obs.shape[0]
    degree = [0] * N
    print("N is", N)
    G = nx.Graph()
#    print("A_obs", _A_obs)
    for row in range(N):
        tmp = 0
        for col in range(row + 1, N):
            if _A_obs[row,col] > 0:
                G.add_edge(row, col, weight = 1)
                G.add_edge(col, row, weight = 1)
                tmp += 1
        degree.append(tmp)
#    print("degree", degree)
#    plt.hist(collections.Counter(degree), bins='auto')

#    plt.show()
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.figure(figsize=(12, 8))
    plt.loglog(degrees[:], degree_freq[:],'go-')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Log Log degree distribution in synthetic dataset')
    plt.show()

        # n, bins, patches = plt.hist(x=gene_chem_edge_list, bins='auto', color='#0504aa',
        #                     alpha=0.7)
        # plt.grid(axis='y', alpha=0.75)
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # plt.title('My Very Own Histogram')
        # plt.text(23, 45, r'$\mu=15, b=3$')
        # maxfreq = n.max()
        # Set a clean upper y-axis limit.
        # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
