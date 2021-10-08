import warnings

warnings.filterwarnings('ignore')

import pickle
import numpy as np
import scipy.sparse as sp
import random
import torch
import os

from cell.utils import link_prediction_performance
from cell.cell import Cell, EdgeOverlapCriterion, LinkPredictionCriterion
from cell.graph_statistics import compute_graph_statistics

from scipy.sparse.csgraph import csgraph_from_dense, connected_components


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


def dataSplit(labeled_ratio, train_ratio, noise_ratio, seed=1111):
    fold_path = '../data/'
    matrix_filename = 'cora_ml.npz'

    symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
    _A_obs, _X_obs, _z_obs = load_npz(fold_path + matrix_filename)

    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1

    lcc = largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]

    _N = _A_obs.shape[0]
    A = sp.tril(_A_obs).tocsr()
    A.eliminate_zeros()
    # labeled_ratio = 0.7
    _E, _N = A.nnz, A.shape[0]

    # labeled edges
    _E = int(_E * labeled_ratio)
    print("labeled edges", _E)
    # train ratio = positive number in train vs positive number in test
    # in fact, test have 1:1 positive number plus negative number
    _size_train = int(_E * train_ratio)
    _size_test = _E - _size_train

    print("size of non zero of cora-ml = ", A.nnz)

    # step 1: sampling enough negative edges for later use
    negative_edges_file_name = "sampling_enough_negative_edges_from_%s" % (matrix_filename)
    _negative_sampling_total = _E * 10
    sampling_zeros = []
    if os.path.exists(fold_path + negative_edges_file_name):
        [sampling_zeros] = load_variable(fold_path + negative_edges_file_name)
    else:
        while len(sampling_zeros) < _negative_sampling_total:
            r, c = np.random.randint(0, _N, 2)
            if A.todense().item(r, c) == 0 and r > c and (r, c) not in sampling_zeros:
                sampling_zeros.append((r, c))
            save_tofile([sampling_zeros], fold_path + negative_edges_file_name)

    # Step 2: test zeros
    test_zeros = []
    other_zeros_for_unlabeled = []
    print(" test size", _size_test)
    test_zeros = random.Random(seed).sample(sampling_zeros, _size_test)
    print(" test zeros", test_zeros[:4], "sampling zero", sampling_zeros[:3])
    for [i, j] in sampling_zeros:
        # print("i j is", i, j)
        if (i, j) in test_zeros:
            continue
        other_zeros_for_unlabeled.append((i, j))
    assert len(other_zeros_for_unlabeled) + _size_test == len(sampling_zeros)
    test_zeros = random.Random(seed).sample(sampling_zeros, _size_test)

    # Step 3: test ones
    i, j = A.nonzero()
    all_ones = np.column_stack((i, j))
    print("size of all ones = ", len(all_ones))

    test_ones = random.Random(seed).sample(all_ones.tolist(), _size_test)
    print("size of test one = ", len(test_ones), "size of test zero = ", len(test_zeros))

    train_ones = []
    for [i, j] in all_ones:
        if [i, j] in test_ones:
            continue
        train_ones.append((i, j))
    print("train ones = ", len(train_ones))

    # unlabeled_ones in other words: zeros as noise ones
    unlabeled_ones = A.nnz - _E
    print("unlabeled ones = ", unlabeled_ones)
    _total = unlabeled_ones / (1 - noise_ratio)
    noise_zeros_size = int(_total * noise_ratio)

    unlabeled_zeros = random.Random(seed).sample(other_zeros_for_unlabeled, noise_zeros_size)
    for (i, j) in unlabeled_zeros:
        train_ones.append((i, j))

    print("train ones add noise edges = ", len(train_ones))

    test_zeros = np.array(test_zeros)
    test_zeros = symmetrize(test_zeros)
    test_ones = np.array(test_ones)
    test_ones = symmetrize(test_ones)
    train_ones = np.array(train_ones)
    train_ones = symmetrize(train_ones)

    train_graph = sp.coo_matrix((np.ones(len(train_ones)), (train_ones[:, 0], train_ones[:, 1]))).tocsr()

    print("size of train graph", train_graph.nnz)

    return train_graph, test_ones, test_zeros


if __name__ == '__main__':
    # labeled_ratio, train_ratio, noise_ratio
    labeled_ratio = 0.5
    train_ratio = 0.5
    # train_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    noise_ratio_list = [0.9, 0.875, 0.83, 0.75, 0.5]
    # noise_ratio_list = [0.5]
    noise_ratio = 0.5

    H_list = [9]
    eval_fre = 2

    for noise_ratio in noise_ratio_list:
        res_file = "evl_data_noise_ratio_%s_.txt" % (str(noise_ratio))
        train_graph, test_ones, test_zeros = dataSplit(labeled_ratio, train_ratio, noise_ratio)
        print(res_file)
        model = Cell(A=train_graph, H=9,
                     callbacks=[LinkPredictionCriterion(invoke_every=eval_fre,
                                                        val_ones=test_ones,
                                                        val_zeros=test_zeros,
                                                        max_patience=200,
                                                        resfile=res_file)])
        model.train(steps=100, optimizer_fn=torch.optim.Adam,
                    optimizer_args={'lr': 0.05,
                                    'weight_decay': 1e-6})
