import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from matplotlib import pyplot as plt

def save_tofile(var_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)
        
def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list
    
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
    
def plot_degree(degree_file):
    
    if os.path.isfile(degree_file):
        [degree] = load_variable(degree_file)
        
        plt.hist(collections.Counter(degree), bins='auto')
        plt.title('Histogram of degree')
        plt.show()


if __name__ == "__main__":
    degree_file = "cora_degree_distribution.pkl"
    input_file = 'cora_ml.npz'
    _A_obs, _X_obs, _z_obs= load_npz(input_file)
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    
    print("size:",_A_obs.shape[0],"edges", _A_obs.getnnz())
    lcc = largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]
    graph_size = _A_obs.shape[0]
    print(" graph size ", graph_size)
    degree = [0] * graph_size
    _A_obs = _A_obs.todense()
    
#    if os.path.isfile(degree_file):
#        [degree] = load_variable(degree_file)
#    else:
#        for row in range(graph_size):
#            tmp = 0
#            for col in range(row+1,graph_size):
#                if _A_obs.item(row, col) > 0:
#                    tmp += 1
#            degree.append(tmp)
#        save_tofile([degree], degree_file)
#    plot_degree(degree_file)

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
