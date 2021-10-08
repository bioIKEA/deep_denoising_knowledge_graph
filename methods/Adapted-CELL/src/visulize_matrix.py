import matplotlib.pyplot as plt
import networkx as nx
import pickle
import scipy.sparse as sp
from cell import utils
import numpy as np
from scipy.sparse import csr_matrix
import csv

def load_variable(filename):
    with open(filename, 'rb') as f:
        [var_list] = pickle.load(f)
    return var_list

def save_to_file(var_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)

score_file = "score_matrix_visualization.pkl"
adj_file = '../data/covid_19/covid_19_sp.npz'
test_index_file = "../data/covid_19/test_dataset/test_idx.pkl"

sp_adj_matrix = sp.load_npz(adj_file)
adj_matrix = sp_adj_matrix.todense()
score_matrix = load_variable(score_file)
test_idx = load_variable(test_index_file)

#pred_graph = utils.graph_from_scores(score_matrix, 18)
#print("predict graph", pred_graph)
#exit(-2)
gene_chem_node_idx = set()
gene_dise_node_idx = set()
chem_dise_node_idx = set()

for test_id in test_idx[:258]:
    gene_chem_node_idx.add(test_id[0])
    gene_chem_node_idx.add(test_id[1])

for _id in test_idx[258:358]:
    gene_dise_node_idx.add(_id[0])
    gene_dise_node_idx.add(_id[1])

for test_part_id in test_idx[358:]:
    chem_dise_node_idx.add(test_part_id[0])
    chem_dise_node_idx.add(test_part_id[1])

#node_id = set()
#for item in test_idx:
#    node_id.add(item[0])
#    node_id.add(item[1])

#for (x, y, val) in test_idx:


#for (x,y,val) in test_idx[358:]:
#    if val == 1:
#        adj_matrix[x,y] = 10
#        adj_matrix[y,x] = 10
#    else:
#        adj_matrix[x,y] = 0
#        adj_matrix[y,x] = 0

#node_id = list(sorted(node_id))
#node_id = list(sorted(chem_dise_node_idx))
#filter_score = score_matrix[node_id,:][:,node_id]
#print("origin matrix nnz is", np.count_nonzero(adj_matrix))

pred_graph = utils.graph_from_scores(score_matrix, 258)
#print("predict graph", pred_graph, pred_graph.getnnz())
#filter_matrix = adj_matrix[node_id,:][:, node_id]
print("print origin matrix")
matrix_file = "ori_matrix.tsv"
with open(matrix_file, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for row in range(adj_matrix.shape[0]):
        for col in range(adj_matrix.shape[1]):
            if row >=0 and row <=6139 and col >= 6140 and col <= 10480 and adj_matrix[row, col] > 0:
                tsv_writer.writerow([row, col, 10])
                adj_matrix[row,col] = 10
        #if row in gene_dise_node_idx and col in gene_dise_node_idx:
            elif row >=0 and row <=6139 and col >= 10481 and col <=23586 and adj_matrix[row, col] > 0:
                tsv_writer.writerow([row, col, 20])
                adj_matrix[row,col] = 20
        #if row in chem_dise_node_idx and col in chem_dise_node_idx:
            elif row >= 6140 and row <= 10480 and col >= 10481 and col <= 23586 and adj_matrix[row, col] > 0:
                tsv_writer.writerow([row, col, 30])
                adj_matrix[row, col] = 30
            else:
                adj_matrix[row, col] = 0

sparse_matrix = csr_matrix(adj_matrix)
sparse_matrix.eliminate_zeros()
#print("predict graph non zero", pred_graph.nonzero())
cnt_gene_chem = 0
cnt_gene_dise = 0
cnt_chem_dise = 0
_x, _y = 0, 0
for (row, col) in zip(*pred_graph.nonzero()):
    #if row in gene_chem_node_idx and col in gene_chem_node_idx:
    if row >=0 and row <=6139 and col >= 6140 and col <= 10480:
        cnt_gene_chem += 1
        pred_graph[row,col] = 10
    #if row in gene_dise_node_idx and col in gene_dise_node_idx:
    if row >=0 and row <=6139 and col >= 10481 and col <=23586:
        cnt_gene_dise += 1
        pred_graph[row,col] = 20
    #if row in chem_dise_node_idx and col in chem_dise_node_idx:
    if row >= 6140 and row <= 10480 and col >= 10481 and col <= 23586:
        cnt_chem_dise += 1
        _x, _y = row, col
        pred_graph[row, col] = 30
print("gene-chem = ", cnt_gene_chem, "gene-dise = ", cnt_gene_dise, "chem-dise = ", cnt_chem_dise)
    #print(row, col)
#print("non zeros = ",np.count_nonzero(filter_matrix))
print("pred", pred_graph)
print("value of ", pred_graph[_x, _y])
#:wqprint("score", pred_graph)
#print("matrix", filter_matrix)
#exit(-1)
#score_normaliz = filter_score / filter_score.sum(axis=1, keepdims=1)
#matrix_normalize = filter_matrix / filter_matrix.sum(axis=1)

#for i in range(95):
#    tmp = 0
#    for j in range(95):
#        tmp += matrix_normalize.item(i,j)
#    print(tmp)

save_to_file(sparse_matrix, "vis_matrix.pkl")
save_to_file(pred_graph, "vis_score.pkl")
exit(-1)
