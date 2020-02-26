from __future__ import print_function

import argparse
import time
from collections import OrderedDict
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from train_data import GraphDataSet
from captum.attr import IntegratedGradients

def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())


def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x


class Message_Passing(nn.Module):
    def forward(self, x, adjacency_matrix):
        neighbor_nodes = torch.bmm(adjacency_matrix, x)
        logging.debug('neighbor message\t', neighbor_nodes.size())
        x = x + neighbor_nodes
        logging.debug('x shape\t', x.size())
        return x


class GraphModel(nn.Module):
    def __init__(self, max_node_num, atom_attr_dim, latent_dim):
        super(GraphModel, self).__init__()

        self.max_node_num = max_node_num
        self.atom_attr_dim = atom_attr_dim
        self.latent_dim = latent_dim


        self.graph_modules = nn.Sequential(OrderedDict([
            ('message_passing_0', Message_Passing()),
            ('dense_0', nn.Linear(self.atom_attr_dim, 50)),
            ('activation_0', nn.Sigmoid()),
            ('message_passing_1', Message_Passing()),
            ('dense_1', nn.Linear(50, self.latent_dim)),
            ('activation_1', nn.Sigmoid()),
        ]))

        self.fully_connected = nn.Sequential(
            nn.Linear(self.max_node_num * self.latent_dim + 1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        return

    def forward(self, node_attr_matrix, adjacency_matrix, t_matrix):
        node_attr_matrix = node_attr_matrix.float()
        adjacency_matrix = adjacency_matrix.float()
        x = node_attr_matrix
        logging.debug('shape\t', x.size())

        for (name, module) in self.graph_modules.named_children():
            if 'message_passing' in name:
                x = module(x, adjacency_matrix=adjacency_matrix)
            else:
                x = module(x)

        # Before flatten, the size should be [Batch size, max_node_num, latent_dim]
        logging.debug('size of x after GNN\t', x.size())
        # After flatten is the graph representation
        x = x.view(x.size()[0], -1)
        logging.debug('size of x after GNN\t', x.size())

        # Concatenate [x, t]
        x = torch.cat((x, t_matrix), 1)

        x = self.fully_connected(x)
        return x


def train(model, data_loader):
    criterion = nn.MSELoss()
    model.train()

    print()
    print("*** Training started! ***")
    print()

    for epoch in range(epochs):
        total_macro_loss = []
        total_mse_loss = []
        if epoch % (epochs / 10) == 0 or epoch == epochs-1:
            torch.save(model.state_dict(), '{}/checkpoint_{}.pth'.format(checkpoint_dir, epoch))
            print('Epoch: {}, Checkpoint saved!'.format(epoch))
        else:
            print('Epoch: {}'.format(epoch))

        train_start_time = time.time()

        for batch_id, (adjacency_matrix, node_attr_matrix, t_matrix, label_matrix) in enumerate(data_loader):
            adjacency_matrix = tensor_to_variable(adjacency_matrix)
            node_attr_matrix = tensor_to_variable(node_attr_matrix)
            t_matrix = tensor_to_variable(t_matrix)
            label_matrix = tensor_to_variable(label_matrix)

            optimizer.zero_grad()

            y_pred = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix, t_matrix=t_matrix)
            loss = criterion(y_pred, label_matrix)
            total_macro_loss.append(macro_avg_err(y_pred, label_matrix).item())
            total_mse_loss.append((loss.item()))
            loss.backward()
            optimizer.step()

        total_macro_loss = np.mean(total_macro_loss)
        total_mse_loss = np.mean(total_mse_loss)
        scheduler.step(total_mse_loss)
        train_end_time = time.time()
        test_loss_epoch = test(model, test_dataloader, 'Test', False)
        print('Train time: {:.3f}s. Training loss is {}. Test loss is {}'.format(train_end_time - train_start_time,
                                                                                 total_macro_loss, test_loss_epoch))

def test(model, data_loader, test_or_tr, printcond):
    model.eval()
    if data_loader is None:
        return None, None

    y_label_list, y_pred_list, total_loss = [], [], 0

    for batch_id, (adjacency_matrix, node_attr_matrix, t_matrix, label_matrix) in enumerate(data_loader):
        adjacency_matrix = tensor_to_variable(adjacency_matrix)
        node_attr_matrix = tensor_to_variable(node_attr_matrix)
        t_matrix = tensor_to_variable(t_matrix)
        label_matrix = tensor_to_variable(label_matrix)

        y_pred = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix, t_matrix=t_matrix)

        y_label_list.extend(variable_to_numpy(label_matrix))
        y_pred_list.extend(variable_to_numpy(y_pred))

    norm = np.load('data/norm.npz', allow_pickle=True)['norm']
    label_mean, label_std = norm[0], norm[1]

    y_label_list = np.array(y_label_list) * label_std + label_mean
    y_pred_list = np.array(y_pred_list) * label_std + label_mean

    total_loss = macro_avg_err(y_pred_list, y_label_list)

    length, w = np.shape(y_label_list)
    if printcond:
        print()
        print('{} Set Predictions: '.format(test_or_tr))
        for i in range(0, length):
            print('True:{}, Predicted: {}'.format(y_label_list[i], y_pred_list[i]))

    return total_loss

def get_data():
    indices = np.load(idx_path, allow_pickle=True)['indices']
    test_idx = indices[running_index]
    train_idx = indices[[i for i in range(folds) if i != running_index]]
    train_idx = [item for sublist in train_idx for item in sublist]

    dataset = GraphDataSet()
    train_data = torch.utils.data.DataLoader(dataset, batch_size=given_args.batch_size,
                                                   sampler=SubsetRandomSampler(train_idx))
    test_data = torch.utils.data.DataLoader(dataset, batch_size=given_args.batch_size,
                                                  sampler=SubsetRandomSampler(test_idx))

    return train_data, test_data

def mse(Y_prime, Y):
    return np.mean((Y_prime - Y) ** 2)

def macro_avg_err(Y_prime, Y):
    if type(Y_prime) is np.ndarray:
        return np.abs(np.sum((Y - Y_prime)) / np.sum(Y))
    return torch.abs(torch.sum(Y - Y_prime) / torch.sum(Y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_node_num', type=int, default=737)
    parser.add_argument('--atom_attr_dim', type=int, default=5)
    parser.add_argument('--latent_dim', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/')
    parser.add_argument('--running_index', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--idx_path', type=str, default='refined_data/indices.npz')
    given_args = parser.parse_args()
    epochs = given_args.epochs
    max_node_num = given_args.max_node_num
    atom_attr_dim = given_args.atom_attr_dim
    latent_dim = given_args.latent_dim
    checkpoint_dir = given_args.checkpoint
    running_index = given_args.running_index
    idx_path = given_args.idx_path
    folds = given_args.folds

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.manual_seed(given_args.seed)

    # Define the model
    model = GraphModel(max_node_num=max_node_num, atom_attr_dim=atom_attr_dim, latent_dim=latent_dim)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=given_args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20,
                                                     min_lr=given_args.min_learning_rate, verbose=True)

    # get the data
    train_dataloader, test_dataloader = get_data()

    # train the mode
    train(model, train_dataloader)

    # predictions on the entire training and test datasets
    train_loss = test(model, train_dataloader, 'Training', True)
    test_loss = test(model, test_dataloader, 'Test', True)

    ig = IntegratedGradients(model)

    # attributions, approximation_error = ig.attribute((input1, input2),
    #                                                  method='gausslegendre',
    #                                                  return_convergence_delta=True)

    print()
    print('--------------------')
    print()
    print("Test Error: {}".format(test_loss))
    print("Training Error: {}".format(train_loss))

