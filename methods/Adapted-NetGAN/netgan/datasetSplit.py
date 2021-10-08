import csv
import pickle
import os
import collections
from collections import defaultdict
import random


def getNodeID(filename):
    """
    The first column is other node term
    the third column is disease terms
    Building a dic with pmc_id as key, texts as value list
    if no pmc_id, skip
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


if __name__ == "__main__":

    print("litcovid dataset:")
    chem_dise_litcovid_filename = "lit_Chem_Dise_Query.tsv"
    chemical_litcovid_dict, chem_disease_litcovd_dict = getNodeID(
        chem_dise_litcovid_filename)
    print("Chem-Dise:", len(chemical_litcovid_dict.keys()),
          len(chem_disease_litcovd_dict.keys()))

    gene_dise_litcovid_filename = "lit_Gene_Dise_Query.tsv"
    gene_litcovid_dict, gene_disease_litcovd_dict = getNodeID(
        gene_dise_litcovid_filename)
    print("Gene-Dise:", len(gene_litcovid_dict.keys()),
          len(gene_disease_litcovd_dict.keys()))

    chem_gene_litcovid_filename = "lit_Gene_Chem_Query.tsv"
    all_gene_litcovid_dict, all_chem_litcovd_dict = getNodeID(
        chem_gene_litcovid_filename)
    print("Gene-Chem:", len(all_gene_litcovid_dict.keys()),
          len(all_chem_litcovd_dict.keys()))

    filtered_gene_litcovd_dict = defaultdict(list)
    filtered_chem_litcovd_dict = defaultdict(list)
    for key in gene_litcovid_dict:
        if key in all_gene_litcovid_dict:
            filtered_gene_litcovd_dict[key] = gene_litcovid_dict[key]
    for key in chemical_litcovid_dict:
        if key in all_chem_litcovd_dict:
            filtered_chem_litcovd_dict[key] = chemical_litcovid_dict[key]

    print("After filter:", len(filtered_gene_litcovd_dict.keys()),
          len(filtered_chem_litcovd_dict.keys()))

    print("pubtator dataset:")
    chem_dise_pub_filename = "Pub_Chem_Dise_Query.tsv"
    chemical_pub_dict, chem_disease_pub_dict = getNodeID(
        chem_dise_pub_filename)
    print("Chem-Dise:", len(chemical_pub_dict.keys()),
          len(chem_disease_pub_dict.keys()))

    gene_dise_pub_filename = "Pub_Gene_Dise_Query.tsv"
    gene_pub_dict, gene_disease_pub_dict = getNodeID(gene_dise_pub_filename)
    print("Gene-Dise:", len(gene_pub_dict.keys()),
          len(gene_disease_pub_dict.keys()))

    chem_gene_pub_filename = "Pub_Gene_Chem_Query.tsv"
    all_gene_pub_dict, all_gene_chem_pub_dict = getNodeID(
        chem_gene_pub_filename)
    print("Gene-Chem:", len(all_gene_pub_dict.keys()),
          len(all_gene_chem_pub_dict.keys()))

    filtered_gene_pub_dict = defaultdict(list)
    filtered_chem_pub_dict = defaultdict(list)
    for key in gene_pub_dict:
        if key in all_gene_pub_dict:
            filtered_gene_pub_dict[key] = gene_pub_dict[key]
    for key in chemical_pub_dict:
        if key in all_gene_chem_pub_dict:
            filtered_chem_pub_dict[key] = chemical_pub_dict[key]

    print("After filter:", len(filtered_gene_pub_dict.keys()),
          len(filtered_chem_pub_dict.keys()))

    dict_list = [chemical_litcovid_dict, chem_disease_litcovd_dict, gene_litcovid_dict, gene_disease_litcovd_dict, filtered_gene_litcovd_dict, filtered_chem_litcovd_dict,
                 chemical_pub_dict, chem_disease_pub_dict, gene_pub_dict, gene_disease_pub_dict, filtered_gene_pub_dict, filtered_chem_pub_dict]

    save_tofile([dict_list], 'idx_to_text.pkl')

    # list_dict_filename = "node_list_dict.pkl"
    # node_list = []
    # node_dict = dict()
    # if os.path.isfile(list_dict_filename):
    #     [node_list, node_dict] = load_variable(list_dict_filename)
    #
    # adj_matrx_graph_filename = "adj_matrix.pkl"
    # graph_matrix = None
    #
    # if os.path.isfile(adj_matrx_graph_filename):
    #     [graph_matrix] = load_variable(adj_matrx_graph_filename)
    #
    # graph_size = len(graph_matrix)

    # get the all edges of each type as a list
    #
    # gene_chem_indexs_list = []
    # gene_dise_indexs_list = []
    # chem_dise_indexs_list = []
    # for row in range(graph_size):
    #     for col in range(row+1, graph_size):
    #         if graph_matrix[row][col] > 0:
    #             if row >= 0 and row <= 6139 and col >= 6140 and col <= 10480:
    #                 gene_chem_indexs_list.append((row, col))
    #             elif row >= 0 and row <= 6139 and col >= 10481 and col <= 23586:
    #                 gene_dise_indexs_list.append((row, col))
    #             elif row >= 6140 and row <= 10480 and col >= 10481 and col <= 23586:
    #                 chem_dise_indexs_list.append((row, col))
    #
    # print("verify gene-chem list", len(gene_chem_indexs_list))
    # print("verify gene-dise list", len(gene_dise_indexs_list))
    # print("verify chem-dise list", len(chem_dise_indexs_list))
    #
    # seed = 1212
    #
    # val_gene_chem_indexs_list = random.Random(
    #     seed).sample(gene_chem_indexs_list, 350)
    # val_gene_dise_indexs_list = random.Random(
    #     seed).sample(gene_dise_indexs_list, 100)
    # val_chem_dise_indexs_list = random.Random(
    #     seed).sample(chem_dise_indexs_list, 50)
    #
    # # print(val_chem_dise_indexs_list[:10])
    # gene_chem_filename = "val_gene_chem_keyword_pair_changed.tsv"
    # gene_dise_filename = "val_gene_dise_keyword_pair_changed.tsv"
    # chem_dise_filename = "val_chem_dise_keyword_pair_changed.tsv"
    #
    # with open(gene_chem_filename, 'wt') as out_file:
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     for (x, y) in val_gene_chem_indexs_list:
    #         # print(x,y)
    #         # print(set(getkeyText(dict_list, node_list[x])), set(getkeyText(dict_list, node_list[y])))
    #         tsv_writer.writerow([set(getkeyText(dict_list, node_list[x])), set(
    #             getkeyText(dict_list, node_list[y]))])
    #
    # with open(gene_dise_filename, 'wt') as out_file:
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     for (x, y) in val_gene_dise_indexs_list:
    #         # print(x,y)
    #         # print(set(getkeyText(dict_list, node_list[x])), set(getkeyText(dict_list, node_list[y])))
    #         tsv_writer.writerow([set(getkeyText(dict_list, node_list[x])), set(
    #             getkeyText(dict_list, node_list[y]))])
    #
    # with open(chem_dise_filename, 'wt') as out_file:
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     for (x, y) in val_chem_dise_indexs_list:
    #         # print(x,y)
    #         # print(set(getkeyText(dict_list, node_list[x])), set(getkeyText(dict_list, node_list[y])))
    #         tsv_writer.writerow([set(getkeyText(dict_list, node_list[x])), set(
    #             getkeyText(dict_list, node_list[y]))])

    # test = "test.tsv"
    # test_dataset_file = "../data/covid_19/test_dataset/test_idx.pkl"
    # [test_dataste_indexs] = load_variable(test_dataset_file)
    #
    # with open(test, 'wt') as out_file:
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     for (x, y, z) in test_dataste_indexs:
    #         # print(x,y)
    #         # print(set(getkeyText(dict_list, node_list[x])), set(getkeyText(dict_list, node_list[y])))
    #         val = "T" if z == 1 else "F"
    #         tsv_writer.writerow([val, set(getkeyText(dict_list, node_list[x])), set(
    #             getkeyText(dict_list, node_list[y]))])