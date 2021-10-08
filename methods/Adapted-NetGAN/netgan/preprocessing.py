"""
InputFiles:
../data/covid_19/*.tsv

OutputFiles: (adjacent edge sparse matrix)
../data/covid_19.npz
"""

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import csgraph_from_dense
import utils
import csv
import pickle
import os
import collections
from collections import defaultdict
import glob
import random


def getNodeID(filename):
    """
    The first column is other node term
    the third column is disease terms
    Building a dic with pmc_id as key, texts as value list
    if no pmc_id, using text vlaue as the same for key and value
    """
    # dict_1 for chem, or gene
    my_dict_1 = defaultdict(list)
    # dict_2 for disease
    my_dict_2 = defaultdict(list)

    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            items = "".join(row).split('\t')
            if line_count == 0:
                pass
                # print("column name",row)
            else:
                # if pmc_id is absent, use lower case text as id
                if items[0] == '-' or items[0] == "None":
                    pass
                    # my_dict_1[items[1].lower()].append(items[1].lower())
                else:
                    my_dict_1[items[0]].append(items[1])
                if items[2] == '-' or items[2] == "None":
                    pass
                else:
                    my_dict_2[items[2]].append(items[3])
            line_count += 1
    return (my_dict_1, my_dict_2)


def create_node_list_and_matrix(fold_path, output_filename):
    print("litcovid dataset:")
    chem_dise_litcovid_filename = fold_path + "lit_Chem_Dise_Query.tsv"
    chemical_litcovid_dict, chem_disease_litcovd_dict = getNodeID(chem_dise_litcovid_filename)
    print("Chem-Dise:", len(chemical_litcovid_dict.keys()), len(chem_disease_litcovd_dict.keys()))

    gene_dise_litcovid_filename = fold_path + "lit_Gene_Dise_Query.tsv"
    gene_litcovid_dict, gene_disease_litcovd_dict = getNodeID(gene_dise_litcovid_filename)
    print("Gene-Dise:", len(gene_litcovid_dict.keys()), len(gene_disease_litcovd_dict.keys()))

    chem_gene_litcovid_filename = fold_path + "lit_Gene_Chem_Query.tsv"
    all_gene_litcovid_dict, all_chem_litcovd_dict = getNodeID(chem_gene_litcovid_filename)
    print("Gene-Chem:", len(all_gene_litcovid_dict.keys()), len(all_chem_litcovd_dict.keys()))

    filtered_gene_litcovd_dict = defaultdict(list)
    filtered_chem_litcovd_dict = defaultdict(list)
    for key in gene_litcovid_dict:
        if key in all_gene_litcovid_dict:
            filtered_gene_litcovd_dict[key] = gene_litcovid_dict[key]
    for key in chemical_litcovid_dict:
        if key in all_chem_litcovd_dict:
            filtered_chem_litcovd_dict[key] = chemical_litcovid_dict[key]

    print("After filter:", len(filtered_gene_litcovd_dict.keys()), len(filtered_chem_litcovd_dict.keys()))

    print("pubtator dataset:")
    chem_dise_pub_filename = fold_path + "Pub_Chem_Dise_Query.tsv"
    chemical_pub_dict, chem_disease_pub_dict = getNodeID(chem_dise_pub_filename)
    print("Chem-Dise:", len(chemical_pub_dict.keys()), len(chem_disease_pub_dict.keys()))

    gene_dise_pub_filename = fold_path + "Pub_Gene_Dise_Query.tsv"
    gene_pub_dict, gene_disease_pub_dict = getNodeID(gene_dise_pub_filename)
    print("Gene-Dise:", len(gene_pub_dict.keys()), len(gene_disease_pub_dict.keys()))

    chem_gene_pub_filename = fold_path + "Pub_Gene_Chem_Query.tsv"
    all_gene_pub_dict, all_gene_chem_pub_dict = getNodeID(chem_gene_pub_filename)
    print("Gene-Chem:", len(all_gene_pub_dict.keys()), len(all_gene_chem_pub_dict.keys()))

    filtered_gene_pub_dict = defaultdict(list)
    filtered_chem_pub_dict = defaultdict(list)
    for key in gene_pub_dict:
        if key in all_gene_pub_dict:
            filtered_gene_pub_dict[key] = gene_pub_dict[key]
    for key in chemical_pub_dict:
        if key in all_gene_chem_pub_dict:
            filtered_chem_pub_dict[key] = chemical_pub_dict[key]

    print("After filter:", len(filtered_gene_pub_dict.keys()), len(filtered_chem_pub_dict.keys()))

    # gene node list
    lit_gene_node = []
    pub_gene_node = []
    for key in gene_litcovid_dict:
        lit_gene_node.append(key)
    for key in filtered_gene_litcovd_dict:
        lit_gene_node.append(key)
    print("lit gene:", len(set(lit_gene_node)))

    for key in gene_pub_dict:
        pub_gene_node.append(key)
    for key in filtered_gene_pub_dict:
        pub_gene_node.append(key)
    print("pub gene:", len(set(pub_gene_node)))

    gene_node = list(set(lit_gene_node + pub_gene_node))
    print("----all gene:", len(gene_node))

    # chem node list
    lit_chem_node = []
    pub_chem_node = []
    for key in chemical_litcovid_dict:
        lit_chem_node.append(key)
    for key in filtered_chem_litcovd_dict:
        lit_chem_node.append(key)
    print("lit chem:", len(set(lit_chem_node)))

    for key in chemical_pub_dict:
        pub_chem_node.append(key)
    for key in filtered_chem_pub_dict:
        pub_chem_node.append(key)
    print("pub chem:", len(set(pub_chem_node)))

    chem_node = list(set(lit_chem_node + pub_chem_node))
    print("----all chem:", len(chem_node))

    # dise node list
    lit_dise_node = []
    pub_dise_node = []
    for key in chem_disease_litcovd_dict:
        lit_dise_node.append(key)
    for key in gene_disease_litcovd_dict:
        lit_dise_node.append(key)
    print("lit dise:", len(set(lit_dise_node)))

    for key in chem_disease_pub_dict:
        pub_dise_node.append(key)
    for key in gene_disease_pub_dict:
        pub_dise_node.append(key)
    print("pub dise:", len(set(pub_dise_node)))

    dise_node = list(set(lit_dise_node + pub_dise_node))
    print("----all dise:", len(dise_node))

    # Convert the set as list and saved, also build a dic as map for fast retrieve
    # the location of each mesh term in the list.
    # node list ordered by gene, chemical, and disease

    list_dict_filename = fold_path + output_filename
    node_list = []
    node_dict = dict()
    if os.path.isfile(list_dict_filename):
        [node_list, node_dict] = load_variable(list_dict_filename)
    else:
        node_list = gene_node + chem_node + dise_node
        # print("dupliate", [item for item, count in collections.Counter(node_list).items() if count > 1])

        for idx, key in enumerate(node_list):
            node_dict[key] = idx
        # print("tmp_keys:", len(node_dict.keys()))
        save_tofile([node_list, node_dict], list_dict_filename)

    graph_size = len(node_list)
    print("size of graph node:", graph_size, len(node_dict.keys()))


def build_adj_matrix_Graph(path, node_dict):
    graph_size = len(node_dict.keys())
    print("graph size", graph_size)
    adj_matrix = [[0 for i in range(graph_size)] for j in range(graph_size)]
    edges = 0
    edges_index = []
    for filename in glob.glob(path):
        print("processing file:", filename)
        with open(filename, 'rU') as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                items = "".join(row).split('\t')
                n1, n2, count = -1, -1, None
                if line_count == 0:
                    pass
                else:
                    if items[0] in node_dict:
                        n1 = node_dict[items[0]]
                    else:
                        n1 = -1
                    if items[2] in node_dict:
                        n2 = node_dict[items[2]]
                    else:
                        n2 = -1
                    if n1 != -1 and n2 != -1:
                        if n1 == n2:
                            print(n1, n2, "item =", items)
                            exit(0)
                        count = int(items[4])
                        # print("add item",n1, n2, count,graph_size, filename, items)
                        if adj_matrix[n1][n2] == 0:
                            edges += 1
                            edges_index.append((n1, n2))
                            edges_index.append((n2, n1))
                        adj_matrix[n1][n2] += count
                        adj_matrix[n2][n1] += count
                line_count += 1
    print("edge is:", edges, len(edges_index))
    print("dupliate", [item for item, count in collections.Counter(edges_index).items() if count > 1])
    verify = 0
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[0])):
            if adj_matrix[i][j] > 0:
                verify += 1
    print("verify = ", verify)
    return adj_matrix


def save_tofile(var_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)


def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list


def getkeyText(dict_list, key):
    """
    based on node_list as key, get their corresponding text
    """
    for dic in dict_list:
        if key in dic:
            return dic[key]


def preview_loaded_data(filename):
    sparse_matrix = scipy.sparse.load_npz(filename)
    return sparse_matrix


def checkML_data(filename):
    print(utils.load_npz(filename))


def randome_generate():
    N = 10
    matrix = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            matrix[i][j] = np.random.randint(2)
            matrix[j][i] = matrix[i][j]

    # print(matrix)
    # demo_graph = np.mod(np.random.permutation(size * size).reshape(size, size), 2)
    # # print(demo_graph)
    csr_matrix = csgraph_from_dense(matrix)
    # # print(csr_matrix)
    scipy.sparse.save_npz('../data/test.npz', csr_matrix)


def read_test_dataset(graph_matrix):
    test_dataset_file = "../data/covid_19/test_dataset/test_idx.pkl"
    test_dataste_indexs = []

    graph_size = len(graph_matrix)

    # get the all edges of each type as a list
    gene_chem_indexs_list = []
    gene_dise_indexs_list = []
    chem_dise_indexs_list = []
    for row in range(graph_size):
        for col in range(row + 1, graph_size):
            if graph_matrix[row][col] > 0:
                if row >= 0 and row <= 6139 and col >= 6140 and col <= 10480:
                    gene_chem_indexs_list.append((row, col))
                elif row >= 0 and row <= 6139 and col >= 10481 and col <= 23586:
                    gene_dise_indexs_list.append((row, col))
                elif row >= 6140 and row <= 10480 and col >= 10481 and col <= 23586:
                    chem_dise_indexs_list.append((row, col))

    print("verify gene-chem list", len(gene_chem_indexs_list))
    print("verify gene-dise list", len(gene_dise_indexs_list))
    print("verify chem-dise list", len(chem_dise_indexs_list))

    seed = 1212

    val_gene_chem_indexs_list = random.Random(
        seed).sample(gene_chem_indexs_list, 350)
    val_gene_dise_indexs_list = random.Random(
        seed).sample(gene_dise_indexs_list, 100)
    val_chem_dise_indexs_list = random.Random(
        seed).sample(chem_dise_indexs_list, 50)

    gene_chem_filename = "../data/covid_19/test_dataset/val_gene_chem_keyword_pair_changed.tsv"
    gene_dise_filename = "../data/covid_19/test_dataset/val_gene_dise_keyword_pair_changed.tsv"
    chem_dise_filename = "../data/covid_19/test_dataset/val_chem_dise_keyword_pair_changed.tsv"

    gen_chem_labels = []
    gen_dise_labels = []
    che_dise_labels = []

    with open(gene_chem_filename, 'r') as out_file:
        tsv_reader = csv.reader(out_file)
        for row in tsv_reader:
            label = "".join(row).split('\t')
            val = None
            if label[0] == "T":
                val = 1
            else:
                val = 0
            gen_chem_labels.append(val)
    for (x, y) in zip(val_gene_chem_indexs_list, gen_chem_labels):
        test_dataste_indexs.append((x[0], x[1], y))

    with open(gene_dise_filename, 'r') as out_file:
        tsv_reader = csv.reader(out_file)
        for row in tsv_reader:
            label = "".join(row).split('\t')
            val = None
            if label[0] == "T":
                val = 1
            else:
                val = 0
            gen_dise_labels.append(val)
    for (x, y) in zip(val_gene_dise_indexs_list, gen_dise_labels):
        test_dataste_indexs.append((x[0], x[1], y))


    with open(chem_dise_filename, 'r') as out_file:
        tsv_reader = csv.reader(out_file)
        for row in tsv_reader:
            label = "".join(row).split('\t')
            val = None
            if label[0] == "T":
                val = 1
            else:
                val = 0
            che_dise_labels.append(val)
    for (x, y) in zip(val_chem_dise_indexs_list, che_dise_labels):
        test_dataste_indexs.append((x[0], x[1], y))


    if os.path.isfile(test_dataset_file):
        [test_dataste_indexs] = load_variable(test_dataset_file)
    else:
        save_tofile([test_dataste_indexs], test_dataset_file)

    print("test dataset created", test_dataste_indexs)


if __name__ == '__main__':

    fold_path = '../data/covid_19/'
    list_dict_filename = 'node_list_dict.pkl'
    node_list = []
    node_dict = dict()
    test_dataste_indexs = []

    if not os.path.isfile(fold_path + list_dict_filename):
        create_node_list_and_matrix(fold_path, list_dict_filename)

    [node_list, node_dict] = load_variable(fold_path + list_dict_filename)

    adj_matrx_graph_filename = "adj_matrix.pkl"
    graph_matrix = None
    if not os.path.isfile(fold_path + adj_matrx_graph_filename):
        graph_matrix = build_adj_matrix_Graph('../data/covid_19/*.tsv', node_dict)
        save_tofile([graph_matrix], fold_path + adj_matrx_graph_filename)

    [graph_matrix] = load_variable(fold_path + adj_matrx_graph_filename)
    print("loading matrix succefully with size = ", len(graph_matrix))

    """
    check the maximum degree of the node in graph
    """
    # max_degree = 0
    # max_idx = 0
    # for i in range(len(graph_matrix)):
    #     tmp = 0
    #     for j in range(len(graph_matrix)):
    #         if graph_matrix[i][j] > 0:
    #             tmp += 1
    #     if tmp > max_degree:
    #         max_degree = tmp
    #         max_idx = i
    # max_degree    3635 max_idx = 14878
    # print("max_degree ", max_degree, "max_idx = ", max_idx)'

    max_degree_idx = 14878
    # make the whole trainig set as a connected component
    for i in range(len(graph_matrix)):
        if i == max_degree_idx:
            continue
        elif graph_matrix[max_degree_idx][i] == 0:
            graph_matrix[max_degree_idx][i] = 1
            graph_matrix[i][max_degree_idx] = 1



    # convert to sparse matrix
    csr_matrix = csgraph_from_dense(graph_matrix)
    scipy.sparse.save_npz(fold_path + "covid_19_sp_connected.npz", csr_matrix)
    # # print("sparse matrix", csr_matrix)
    # # csr_matrix = scipy.sparse.csr_matrix(graph_matrix)
    # csr_matrix = csr_matrix + csr_matrix.T
    # csr_matrix[csr_matrix > 1] = 1
    # n_components, _ = scipy.sparse.csgraph.connected_components(csr_matrix)
    # # 774 components in whole graphs
    # print("whole graph", csr_matrix.count_nonzero())
    # print(" whole graph componets = ", n_components)
    #
    # """
    # check test datasets graph connected components
    # """
    # graph_size = len(graph_matrix)
    # print("graph size = ", graph_size)
    # # test_graph = [[0 for i in range(graph_size)] for j in range(graph_size)]
    # # run datasetSplit.py to split train and test dataset
    # # then loading the test dataset by reading three generated
    # if not os.path.isfile(fold_path + 'test_dataset/test_idx.pkl'):
    #     read_test_dataset(graph_matrix)
    # [test_dataste_indexs] = load_variable(fold_path + 'test_dataset/test_idx.pkl')

    # print("test dataset finished with size = ", len(test_dataste_indexs))
    # initial_x, initial_y, _ = test_dataste_indexs[0]
    # test_graph[initial_x][initial_y] = 1
    # test_graph[initial_y][initial_x] = 1
    # for (x, y, val) in test_dataste_indexs[1:]:
    #     test_graph[x][y] = 1
    #     test_graph[y][x] = 1
    #     test_graph[x][initial_x] = 1
    #     test_graph[initial_x][x] = 1
    # csr_test_matrix = csgraph_from_dense(test_graph)
    # csr_test_matrix = csr_test_matrix + csr_test_matrix.T
    # csr_test_matrix[csr_test_matrix > 1] = 1
    # print("test graph", csr_test_matrix.count_nonzero())
    # n_components_test, index_label = scipy.sparse.csgraph.connected_components(csr_test_matrix)
    # print(" test graph componets = ", n_components_test)
    # randome_generate()
    # file1 = '../data/cora_ml.npz'
    # checkML_data(file1)
    # file2 = '../data/test.npz'
    # print(preview_loaded_data(file2))
