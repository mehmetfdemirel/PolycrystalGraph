import argparse
from data import GraphDataSet
import numpy as np
from sklearn.model_selection import KFold
import random

def split_data(num_folds, num_data, random_seed):
    total_graph = 492
    graph_seq = np.arange(1,total_graph+1)
    random.Random(random_seed).shuffle(graph_seq)
    dataset = GraphDataSet(num_data, graph_seq)
    num_of_data = dataset.__len__()
    kf = KFold(n_splits=num_folds, shuffle=True, random_state = random_seed)
    ind = []
    for i, (_, index) in enumerate(kf.split(np.arange(num_of_data))):
        ind.append(index)
    ind = np.asarray(ind, dtype=object)
    return graph_seq, ind

def extract_graph_data(out_file_path, indices, graph_seq):
    np.savez_compressed(out_file_path, indices = indices, graph_seq = graph_seq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--num_data', type=int, default=492)
    parser.add_argument('--random_seed', type=int, default=123)
    given_args = parser.parse_args()
    num_folds = given_args.folds
    num_data = given_args.num_data
    random_seed = given_args.random_seed
    out_file_path = 'indices_and_graphseq.npz'

    print("Output File Path: {}".format(out_file_path))

    graph_seq, indices = split_data(num_folds, num_data, random_seed)
    extract_graph_data(out_file_path, indices = indices, graph_seq = graph_seq)

    print("Data successfully split into {} folds!".format(num_folds))
