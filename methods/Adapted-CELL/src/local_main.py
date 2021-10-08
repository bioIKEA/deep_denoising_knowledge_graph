import warnings

warnings.filterwarnings('ignore')

import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz
import random
import torch

import networkx as nx

from cell.utils import link_prediction_performance
from cell.cell import Cell, EdgeOverlapCriterion, LinkPredictionCriterion
from cell.graph_statistics import compute_graph_statistics

from scipy.sparse.csgraph import csgraph_from_dense


def readFile(adj_matrix, test_idx_file):
    train_graph = load_npz(adj_matrix)
    test_data_idx = None
    with open(test_idx_file, 'rb') as f:
        [test_data_idx] = pickle.load(f)

    return (train_graph, test_data_idx)


def dataSplit(train_graph, is_weight, test_data_idx, test_part, seed):
    """
    Args:
        adj_matrix:
        test_data_idx:
        test_part:
            0: all edges as test part,
            1: gene_chem as test part
            2: chem_dise and gene_dise as test part
    Returns:

    """
    symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))

    dense_adj = train_graph.todense()
    if not is_weight:
        # reset weight all to 1
        dense_adj[dense_adj > 1] = 1

    # remove the test dataset edges
    # for (i, j, val) in test_data_idx:
    #    dense_adj[i, j] = 0
    #    dense_adj[j, i] = 0

    test_ones = []
    test_zeros = []
    if test_part == 0:
        for (i, j, val) in test_data_idx:
            if val == 1:
                test_ones.append([i, j])
            else:
                test_zeros.append([i, j])
        print("size of one = ", len(test_ones), len(test_zeros))
    elif test_part == 1:
        for (i, j, val) in test_data_idx[:258]:
            if val == 1:
                test_ones.append([i, j])
            else:
                test_zeros.append([i, j])
    elif test_part == 2:
        for (i, j, val) in test_data_idx[258:]:
            if val == 1:
                test_ones.append([i, j])
            else:
                test_zeros.append([i, j])

    test_size = min(len(test_ones), len(test_zeros))
    print("test size = ", test_size)
    if len(test_ones) != test_size:
        test_ones = random.Random(seed).sample(test_ones, test_size)
    elif len(test_zeros) != test_size:
        test_zeros = random.Random(seed).sample(test_zeros, test_size)

    assert len(test_zeros) == len(test_ones)

    for (i, j) in test_ones:
        dense_adj[i, j] = 0
        dense_adj[j, i] = 0
    for (i, j) in test_zeros:
        dense_adj[i, j] = 0
        dense_adj[j, i] = 0

    test_ones = np.array(test_ones)
    test_zeros = np.array(test_zeros)

    test_ones = symmetrize(test_ones)
    test_zeros = symmetrize(test_zeros)

    print("size of ones and zeros = ", len(test_ones), len(test_zeros))
    return (csgraph_from_dense(dense_adj), test_ones, test_zeros)


def get_my_loss_fn(SP, lam, SP_limit=None):
    SP = torch.tensor(SP)
    if SP_limit:
        SP = lam * (SP <= SP_limit)
    else:
        SP.fill_diagonal_(1)

    def my_loss_fn(W, A, num_edges):
        d = torch.log(torch.exp(W).sum(dim=-1, keepdims=True))
        not_A = 1 - A
        not_A.fill_diagonal_(0)

        if SP_limit:
            loss = torch.sum((A + SP) * (d - W))
        else:
            loss = torch.sum((A + lam * (1 / torch.exp(SP))) * (d - W))

        loss *= .5 / num_edges
        return loss

    return my_loss_fn

if __name__ == '__main__':
    test_idx_file = '../data/covid_19/test_dataset/test_idx.pkl'

    adj_matrix = '../data/covid_19/covid_19_sp.npz'
    # adj_matrix = '../data/covid_19/covid_19_sp_connected.npz'

    train_graph, test_data_idx = readFile(adj_matrix, test_idx_file)
    print(len(test_data_idx))

    is_weight = False
    test_ones = None
    test_zeros = None
    seed = 5
    test_parts = [2]
    H_list = [20]
    eval_fre = 2
    is_weight_list = [False]
    seeds_list = [0, 26]

    for test_part in test_parts:
        for seed in seeds_list:
            train_graph, test_ones, test_zeros = dataSplit(train_graph, is_weight, test_data_idx, test_part, seed)

            G = nx.from_scipy_sparse_matrix(train_graph)
            B = nx.all_pairs_shortest_path_length(G)
            B_dict = dict(B)
            SP = np.zeros(train_graph.shape)
            for start in B_dict.keys():
                for target in B_dict[start].keys():
                    SP[start, target] = B_dict[start][target]
            my_loss_fn = get_my_loss_fn(SP=SP, lam=0, SP_limit=None)


            for h in H_list:
                weighted = 1 if is_weight else 0
                res_file = "local_loss_weight_%s_H_%s_is_weight_%s_seed_%s_.txt" % (
                str(test_part), str(h), str(weighted), str(seed))
                print(res_file)
                model = Cell(A=train_graph, H=h,loss_fn=my_loss_fn,
                             callbacks=[LinkPredictionCriterion(invoke_every=eval_fre,
                                                                val_ones=test_ones,
                                                                val_zeros=test_zeros,
                                                                max_patience=200,
                                                                resfile=res_file)])
                model.train(steps=150, optimizer_fn=torch.optim.Adam,
                            optimizer_args={'lr': 0.05,
                                            'weight_decay': 1e-6})

