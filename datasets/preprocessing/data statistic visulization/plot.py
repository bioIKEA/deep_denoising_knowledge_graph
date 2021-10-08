import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections

def save_tofile(var_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)

def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list

def plot_edge_hist(edge_distribution_file):
    
    if os.path.isfile(edge_distribution_file):
        [gene_chem_edge_list, gene_dise_edge_list, chem_dise_edge_count, all_edges] = load_variable(edge_distribution_file)
        # print("gene-chem", gene_chem_edge_list)
        # hist, bin_edges = np.histogram(gene_chem_edge_list)
        # print("hist", hist)
        # print("bin_edges", bin_edges)

        # with open('edge_distribution_encoding_byte.pkl', 'w') as f:
        #     [gene_chem_edge_list, gene_dise_edge_list, gene_dise_edge_list]

        # print("min:",min(gene_chem_edge_list), "max:",max(gene_chem_edge_list))
        
        plt.hist(collections.Counter(gene_chem_edge_list), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.title('Histogram of gene-chem edge degree', fontsize=12)
        plt.xlabel('Degree', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.show()

        plt.hist(collections.Counter(gene_dise_edge_list), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.title('Histogram of gene-dise edge degree',fontsize=12)
        plt.xlabel('Degree', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.show()

        plt.hist(collections.Counter(chem_dise_edge_count), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.title('Histogram of chem-dise edge degree', fontsize=12)
        plt.xlabel('Degree', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.show()
        
        plt.hist(collections.Counter(all_edges), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.title('Histogram of degree', fontsize=12)
        plt.xlabel('Degree', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.show()
        
def plot_degree(degree_file):
    
    if os.path.isfile(degree_file):
        [degree] = load_variable(degree_file)
        
        plt.hist(collections.Counter(degree), bins='auto')
        plt.title('Histogram of degree')
        plt.show()


if __name__ == "__main__":
    edge_distribution_file = "edge_distribution.pkl"
#    degree_file = "degree.pkl"
    plot_edge_hist(edge_distribution_file)

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
