# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:27:44 2021

@author: daimi
"""

from __future__ import print_function

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from scipy import sparse

class GraphDataSet_interpretable(Dataset):
    def __init__(self):
        max_node = 300
        features = 5

        for i in range(1, 493):
            # load files
            neighbor_file_path = 'data/structure-{}/neighbor.txt'.format(i)
            feature_file_path = 'data/structure-{}/feature.txt'.format(i)
            property_file_path = 'data/structure-{}/property.txt'.format(i)
            neighbor = np.loadtxt(neighbor_file_path)
            feature = np.loadtxt(feature_file_path)
            proprty = np.loadtxt(property_file_path)

            # feature data manipulation
            feature = np.delete(feature, 0, axis=1)
            feature[:, [3]] = (feature[:, [3]] - np.mean(feature[:, [3]])) / np.std(feature[:, [3]])
            feature[:, [4]] = (feature[:, [4]] - np.mean(feature[:, [4]])) / np.std(feature[:, [4]])

            # normalize the adjaciency matrix
            np.fill_diagonal(neighbor, 1)
            D=np.sum(neighbor,axis=0)
            D_inv=np.diag(np.power(D,-0.5))
            neighbor=np.matmul(D_inv, np.matmul(neighbor,D_inv))
            
            # match dimension to the max dimension for neighbors
            result = np.zeros((max_node, max_node))
            result[:neighbor.shape[0], :neighbor.shape[1]] = neighbor
            neighbor = result

            # match dimension to the max dimension for features
            result = np.zeros((max_node, features))
            result[:feature.shape[0], :feature.shape[1]] = feature
            feature = result

            feature = sparse.csr_matrix(feature)
            neighbor = sparse.csr_matrix(neighbor)

            #delete negative value of proprty
            proprty = proprty[proprty.min(axis=1)>=0,:]

            #get the dimension of proprty
            num_properties, width = np.shape(proprty)

            # independent variable t
            t = np.delete(proprty, 1, axis=1)
            t = t[num_properties-1,:]   # get the field of the last data point associated with this microstructure
            t = t.reshape((1,1))

            # label
            label = np.delete(proprty, 0, axis=1)
            label = label[num_properties-1,:]  # get the target of the last data point associated with this microstructure
            label = label.reshape((1,1))

            # gradient will not change for different field value. So only using one point for interpretation
            multiple_neighbor = [neighbor]
            multiple_feature = [feature]

            if i == 1:
                adjacency_matrix = multiple_neighbor
                node_attr_matrix = multiple_feature
                t_matrix = t
                label_matrix = label
            else:
                adjacency_matrix = np.concatenate((adjacency_matrix, multiple_neighbor))
                node_attr_matrix = np.concatenate((node_attr_matrix, multiple_feature))
                t_matrix = np.concatenate((t_matrix, t))
                label_matrix = np.concatenate((label_matrix, label))

        # normalize the independent variable t matrix
        t_matrix = t_matrix / 10000

        # normalize the label matrix
        label_mean = np.mean(label_matrix)
        label_std = np.std(label_matrix)
        label_matrix = (label_matrix - label_mean) / label_std

        norm = np.array([label_mean, label_std])
        np.savez_compressed('interpretation/Interpretable_norm.npz', norm=norm)

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
