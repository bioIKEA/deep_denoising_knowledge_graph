import csv
import pickle
import os
import collections
from collections import defaultdict
import glob

def build_weighted_graph(node_ID_dict, N1_list, N2_list, Value_list):
    print("initialize adj list")
    adj_list = []
    for index, _ in enumerate(N1_list):
        n1_index = node_ID_dict.get(N1_list[index])
        n2_index = node_ID_dict.get(N2_list[index])
        val = Value_list[index]
        adj_list.append((n1_index, n2_index, val))
    return adj_list
    
def save_tofile(var_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)

def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list


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

def build_adj_matrix_Graph(node_dict):
    graph_size = len(node_dict.keys())
    print("graph size", graph_size)
    adj_matrix = [[0 for i in range(graph_size)] for j in range(graph_size)]
    path = "./*.tsv"
    edges = 0
    edges_index = []
    for filename in glob.glob(path):
        print("processing file:", filename)
        with open(filename, 'rU') as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                items = "".join(row).split('\t')
                n1, n2, count= -1, -1, None
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
                    if n1 != -1 and n2!= -1:
                        if n1==n2:
                            print(n1,n2, "item =",items)
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
    print("dupliate",[item for item, count in collections.Counter(edges_index).items() if count > 1])
    verify = 0
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[0])):
            if adj_matrix[i][j] > 0:
                verify += 1
    print("verify = ", verify)
    return adj_matrix

if __name__ == "__main__":

    print("litcovid dataset:")
    chem_dise_litcovid_filename = "lit_Chem_Dise_Query.tsv"
    chemical_litcovid_dict, chem_disease_litcovd_dict = getNodeID(chem_dise_litcovid_filename)
    print("Chem-Dise:", len(chemical_litcovid_dict.keys()), len(chem_disease_litcovd_dict.keys()))

    gene_dise_litcovid_filename = "lit_Gene_Dise_Query.tsv"
    gene_litcovid_dict, gene_disease_litcovd_dict = getNodeID(gene_dise_litcovid_filename)
    print("Gene-Dise:", len(gene_litcovid_dict.keys()), len(gene_disease_litcovd_dict.keys()))

    chem_gene_litcovid_filename = "lit_Gene_Chem_Query.tsv"
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
    chem_dise_pub_filename = "Pub_Chem_Dise_Query.tsv"
    chemical_pub_dict, chem_disease_pub_dict = getNodeID(chem_dise_pub_filename)
    print("Chem-Dise:", len(chemical_pub_dict.keys()), len(chem_disease_pub_dict.keys()))

    gene_dise_pub_filename = "Pub_Gene_Dise_Query.tsv"
    gene_pub_dict, gene_disease_pub_dict = getNodeID(gene_dise_pub_filename)
    print("Gene-Dise:", len(gene_pub_dict.keys()), len(gene_disease_pub_dict.keys()))

    chem_gene_pub_filename = "Pub_Gene_Chem_Query.tsv"
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

    gene_node = list(set(lit_gene_node+pub_gene_node))
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

    chem_node = list(set(lit_chem_node+pub_chem_node))
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

    dise_node = list(set(lit_dise_node+pub_dise_node))
    print("----all dise:", len(dise_node)) 

    # Convert the set as list and saved, also build a dic as map for fast retrieve 
    # the location of each mesh term in the list.
    # node list ordered by gene, chemical, and disease

    list_dict_filename = "node_list_dict.pkl"
    node_list = []
    node_dict = dict()
    if os.path.isfile(list_dict_filename):
        [node_list, node_dict] = load_variable(list_dict_filename)
    else:
        node_list = gene_node + chem_node + dise_node
        print("dupliate",[item for item, count in collections.Counter(node_list).items() if count > 1])
        
        for idx, key in enumerate(node_list):
            node_dict[key] = idx
        print("tmp_keys:",len(node_dict.keys()))
        save_tofile([node_list, node_dict], list_dict_filename)

    graph_size = len(node_list)
    print("size of graph node:", graph_size, len(node_dict.keys()))

    adj_matrx_graph_filename = "adj_matrix.pkl"
    graph_matrix = None
    
    if os.path.isfile(adj_matrx_graph_filename):
        [graph_matrix] = load_variable(adj_matrx_graph_filename)
    else:
        graph_matrix = build_adj_matrix_Graph(node_dict)
        save_tofile([graph_matrix],adj_matrx_graph_filename)

    total_edge = 0
    
    gene_chem_edge_count = 0
    gene_chem_edge_list = []
    gene_dise_edge_count = 0
    gene_dise_edge_list = []
    chem_dise_edge_count = 0
    chem_dise_edge_list = []
    all_edges_list = []
    #  gene, chem, dise index range
    # [0-6139,6140-10480, 10481-23587]
    
    
    edge_distribution_file = "edge_distribution.pkl"
    if os.path.isfile(edge_distribution_file):
        [gene_chem_edge_list, gene_dise_edge_list, chem_dise_edge_list] = load_variable(edge_distribution_file)
    else:
        for row in range(graph_size):
            gene_chem_edge_count = 0
            gene_dise_edge_count = 0
            chem_dise_edge_count = 0
            total_edge = 0
            for col in range(row+1,graph_size):
                if graph_matrix[row][col] > 0:
                    total_edge += 1
                    if row>=0 and row<=6139 and col >= 6140 and col<=10480:
                        gene_chem_edge_count += 1
                    elif row>=0 and row<=6139 and col >= 10481 and col<=23587:
                        gene_dise_edge_count += 1
#                        gene_dise_edge_list.append(graph_matrix[row][col])
                    elif row>=6140 and row<=10480 and col >= 10481 and col<=23587:
                        chem_dise_edge_count += 1
#                        chem_dise_edge_list.append(graph_matrix[row][col])
            if gene_chem_edge_count > 0:
                gene_chem_edge_list.append(gene_chem_edge_count)
            if gene_dise_edge_count > 0:
                gene_dise_edge_list.append(gene_dise_edge_count)
            if chem_dise_edge_count > 0:
                chem_dise_edge_list.append(chem_dise_edge_count)
            all_edges_list.append(total_edge)
                    
        save_tofile([gene_chem_edge_list, gene_dise_edge_list, chem_dise_edge_list, all_edges_list], edge_distribution_file)

    degree_file = "degree.pkl"
    degree = [0] * graph_size
    if os.path.isfile(degree_file):
        [degree] = load_variable(degree_file)
    else:
        for row in range(graph_size):
            tmp = 0
            for col in range(row+1,graph_size):
                if graph_matrix[row][col] > 0:
                    tmp += 1
            degree.append(tmp)
        save_tofile([degree], degree_file)
    
    print("total edge =", total_edge)
    print("gene-chem = ", gene_chem_edge_count)
    print("gene-dise = ", gene_dise_edge_count)
    print("chem-dise = ", chem_dise_edge_count)
    print("sum", gene_chem_edge_count+gene_dise_edge_count+chem_dise_edge_count)
