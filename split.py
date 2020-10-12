import argparse
from data import GraphDataSet
import numpy as np
from sklearn.model_selection import KFold

def split_data():
    dataset = GraphDataSet()
    num_of_data = dataset.__len__()
    kf = KFold(n_splits=num_folds, shuffle=True)
    ind = []
    for i, (_, index) in enumerate(kf.split(np.arange(num_of_data))):
        np.random.shuffle(index)
        ind.append(index)
    ind = np.asarray(ind, dtype=object)
    return ind

def extract_graph_data(out_file_path, ind):
    np.savez_compressed(out_file_path, indices = ind)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    given_args = parser.parse_args()
    num_folds = given_args.folds
    out_file_path = 'data/indices.npz'
    np.random.seed(given_args.seed)

    print("Output File Path: {}".format(out_file_path))

    indices = split_data()
    extract_graph_data(out_file_path, ind = indices)

    print("Data successfully split into {} folds!".format(num_folds))
