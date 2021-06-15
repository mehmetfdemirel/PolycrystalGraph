# load the data into the dataset
from __future__ import print_function
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class GraphDataSet(Dataset):
    def __init__(self, num_data, graph_seq):
        max_node = 300
        num_features = 5
        for i in range(num_data):
            ind = graph_seq[i]
            # load files
            file_paths = ['data/structure-{}/neighbor.txt'.format(ind), 'data/structure-{}/feature.txt'.format(ind),
                          'data/structure-{}/property.txt'.format(ind)]

            graph_elements = [np.loadtxt(file_paths[0]), np.loadtxt(file_paths[1]), np.loadtxt(file_paths[2])]

            # feature data manipulation
            graph_elements[1] = manipulate_feature(graph_elements[1], max_node, num_features)

            # normalize the adjacency matrix
            graph_elements[0] = normalize_adj(graph_elements[0], max_node)

            # delete data points with negative properties
            graph_elements[2] = graph_elements[2][graph_elements[2].min(axis=1) >= 0, :]
            # get the dimension of proprty
            num_properties, width = np.shape(graph_elements[2])
            # independent variable t, the external field
            t = np.delete(graph_elements[2], 1, axis=1)
            # label, the magnetostriction
            label = np.delete(graph_elements[2], 0, axis=1)

            # change it to the several data points
            multiple_neighbor, multiple_feature = [graph_elements[0] for x in range(num_properties)], \
                                                  [graph_elements[1] for x in range(num_properties)]

                # concatenating the matrices
            if i == 0:
                adjacency_matrix, node_attr_matrix, t_matrix, label_matrix = multiple_neighbor, multiple_feature, t, label
            else:
                adjacency_matrix, node_attr_matrix, t_matrix, label_matrix = np.concatenate((adjacency_matrix, multiple_neighbor)), \
                                                                             np.concatenate((node_attr_matrix, multiple_feature)), \
                                                                             np.concatenate((t_matrix, t)),\
                                                                             np.concatenate((label_matrix, label))

        # normalize the independent variable t matrix
        t_matrix, label_matrix = normalize_t_label(t_matrix, label_matrix)

        self.adjacency_matrix = np.array(adjacency_matrix)
        self.node_attr_matrix = np.array(node_attr_matrix)
        self.t_matrix = np.array(t_matrix)
        self.label_matrix = np.array(label_matrix)

        print('--------------------')
        print('Training Data:')
        print('adjacency matrix:\t', self.adjacency_matrix.shape)
        print('node attribute matrix:\t', self.node_attr_matrix.shape)
        print('t matrix:\t\t', self.t_matrix.shape)
        print('label name:\t\t', self.label_matrix.shape)
        print('--------------------')

    def __len__(self):
        return len(self.adjacency_matrix)

    def __getitem__(self, idx):
        adjacency_matrix = self.adjacency_matrix[idx].todense()
        node_attr_matrix = self.node_attr_matrix[idx].todense()
        t_matrix = self.t_matrix[idx]
        label_matrix = self.label_matrix[idx]

        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        node_attr_matrix = torch.from_numpy(node_attr_matrix)
        t_matrix = torch.from_numpy(t_matrix)
        label_matrix = torch.from_numpy(label_matrix)
        return adjacency_matrix, node_attr_matrix, t_matrix, label_matrix

def normalize_adj(neighbor, max_node):
    np.fill_diagonal(neighbor, 1)  # add the identity matrix
    D = np.sum(neighbor, axis=0)  # calculate the diagnoal element of D
    D_inv = np.diag(np.power(D, -0.5))  # construct D
    neighbor = np.matmul(D_inv, np.matmul(neighbor, D_inv))  # symmetric normalization of adjacency matrix

    # match dimension to the max dimension for neighbors
    result = np.zeros((max_node, max_node))
    result[:neighbor.shape[0], :neighbor.shape[1]] = neighbor
    neighbor = result

    # convert the feature matrix to sparse matrix
    neighbor = sparse.csr_matrix(neighbor)

    return neighbor

def manipulate_feature(feature, max_node, features):
    feature = np.delete(feature, 0, axis=1)  # remove the first column (Grain ID)
    feature[:, [3]] = (feature[:, [3]] - np.mean(feature[:, [3]])) / np.std(
        feature[:, [3]])  # normalize grain size
    feature[:, [4]] = (feature[:, [4]] - np.mean(feature[:, [4]])) / np.std(
        feature[:, [4]])  # normalize number of neighbors

    # match dimension to the max dimension for features
    result = np.zeros((max_node, features))
    result[:feature.shape[0], :feature.shape[1]] = feature
    feature = result

    # convert the feature matrix to sparse matrix
    feature = sparse.csr_matrix(feature)

    return feature

def normalize_t_label(t_matrix, label_matrix):
    t_matrix = t_matrix / 10000
    label_mean = np.mean(label_matrix)
    label_std = np.std(label_matrix)
    label_matrix = (label_matrix - label_mean) / label_std

    # save the mean and standard deviation of label
    norm = np.array([label_mean, label_std])
    np.savez_compressed('norm.npz', norm=norm)

    return t_matrix, label_matrix


def get_data(batch_size, idx_path, validation_index, testing_index, folds, num_data):
    indices = np.load(idx_path, allow_pickle=True)['indices']
    graph_seq = np.load(idx_path, allow_pickle=True)['graph_seq']
    validation_idx = indices[validation_index]
    test_idx = indices[testing_index]
    train_idx = indices[[i for i in range(folds) if i != validation_index and i != testing_index]]
    train_idx = [item for sublist in train_idx for item in sublist]

    dataset = GraphDataSet(num_data, graph_seq)
    train_data = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    validation_data = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_idx))
    test_data = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))
    return train_data, validation_data, test_data
