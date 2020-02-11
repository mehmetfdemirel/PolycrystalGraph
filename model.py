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
            nn.Linear(self.max_node_num * self.latent_dim + 1, 1),
            #nn.Linear(1024, 256),
            #nn.Linear(256, 64),
            #nn.Linear(64, 1)
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
    model.train()

    total_loss = 0
    for batch_id, (adjacency_matrix, node_attr_matrix, t_matrix, label_matrix) in enumerate(data_loader):
        adjacency_matrix = tensor_to_variable(adjacency_matrix)
        node_attr_matrix = tensor_to_variable(node_attr_matrix)
        t_matrix = tensor_to_variable(t_matrix)
        label_matrix = tensor_to_variable(label_matrix)

        optimizer.zero_grad()

        y_pred = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix, t_matrix=t_matrix)
        loss = criterion(y_pred, label_matrix)
        total_loss += MacroAvgRelErr(y_pred, label_matrix) #loss.data

        loss.backward()
        optimizer.step()

    total_loss /= len(data_loader.sampler)
    return total_loss


def MSE(Y_prime, Y):
    return np.mean((Y_prime - Y) ** 2)

def MacroAvgRelErr(Y_prime, Y):
    if type(Y_prime) is np.ndarray:
        return np.abs(np.sum((Y - Y_prime)) / np.sum(Y))
    return torch.abs(torch.sum(Y - Y_prime) / torch.sum(Y))

def test(model, data_loader, fold, test_or_tr, printcond):
    model.eval()
    if data_loader is None:
        return None, None

    y_label_list = []
    y_pred_list = []
    total_loss = 0
    for batch_id, (adjacency_matrix, node_attr_matrix, t_matrix, label_matrix) in enumerate(data_loader):
        adjacency_matrix = tensor_to_variable(adjacency_matrix)
        node_attr_matrix = tensor_to_variable(node_attr_matrix)
        t_matrix = tensor_to_variable(t_matrix)
        label_matrix = tensor_to_variable(label_matrix)

        y_pred = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix, t_matrix=t_matrix)
        total_loss += MacroAvgRelErr(y_pred, label_matrix) #criterion(y_pred, label_matrix)

        y_label_list.extend(variable_to_numpy(label_matrix))
        y_pred_list.extend(variable_to_numpy(y_pred))

    total_loss /= len(data_loader.sampler)

    normalization = np.loadtxt('normalization.in')
    label_mean = normalization[0]
    label_std = normalization[1]

    y_label_list = np.array(y_label_list) * label_std + label_mean
    y_pred_list = np.array(y_pred_list) * label_std + label_mean

    length, w = np.shape(y_label_list)
    if printcond:
        print('')
        if fold != -1:
            print('Fold {} {} Predictions: '.format(fold, test_or_tr))
        else:
            print('Predictions on the Entire Training Set:')
        for i in range(0, length):
            print('True:{}, Predicted: {}'.format(y_label_list[i], y_pred_list[i]))

    if printcond:
        print(' ')
        print('Macro Averaged Error: {}'.format(total_loss))
        if fold != -1:
            print('Fold {} {} Macro Averaged Error: {}'.format(fold, test_or_tr, total_loss))

    return total_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_node_num', type=int, default=737)
    parser.add_argument('--atom_attr_dim', type=int, default=5)
    parser.add_argument('--latent_dim', type=int, default=50)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model/')
    given_args = parser.parse_args()

    epochs = given_args.epoch
    max_node_num = given_args.max_node_num
    atom_attr_dim = given_args.atom_attr_dim
    latent_dim = given_args.latent_dim
    checkpoint_dir = given_args.checkpoint

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.manual_seed(given_args.seed)

    # Define the model
    model = GraphModel(max_node_num=max_node_num, atom_attr_dim=atom_attr_dim, latent_dim=latent_dim)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=given_args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50,
                                                     min_lr=given_args.min_learning_rate, verbose=True)
    criterion = nn.MSELoss()

    # Specify data
    import train_data

    dataset = train_data.GraphDataSet_Adjacent()


    num_of_data = dataset.__len__()
    indices = list(range(num_of_data))
    np.random.shuffle(indices)
    split_fold = 5
    each_bin = num_of_data // split_fold
    print('*********************************************************')
    sum_mse = 0
    ftrain = open("training_losses.txt", "w+")
    ftest = open("test_losses.txt", "w+")
    for fold in range(0, split_fold):
        model.__init__(max_node_num=max_node_num, atom_attr_dim=atom_attr_dim, latent_dim=latent_dim)

        print('')
        print('FOLD {} OF {} STARTED!'.format(fold + 1, split_fold))

        if fold != split_fold - 1:
            test_indices = indices[fold * each_bin:fold * each_bin + each_bin]
        else:
            test_indices = indices[fold * each_bin:num_of_data - 1]

        train_indices = [idx for idx in indices if idx not in test_indices]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=given_args.batch_size, sampler=train_sampler)
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=given_args.batch_size, sampler=test_sampler)
        full_training = torch.utils.data.DataLoader(dataset, batch_size=given_args.batch_size, shuffle=True)

        for epoch in range(epochs):
            print('Epoch: {}'.format(epoch))
            if epoch % (epochs / 10) == 0:
                torch.save(model.state_dict(), '{}/checkpoint_{}.pth'.format(checkpoint_dir, epoch))
                print('Model saved.')
            train_start_time = time.time()
            train_loss = train(model, train_dataloader)
            ftrain.write('{}\n'.format(train_loss))
            scheduler.step(train_loss)
            train_end_time = time.time()
            test_loss = test(model, test_dataloader, fold + 1, 'Test', False)
            print('Train time: {:.3f}s. Training loss is {}. Test loss is {}'.format(train_end_time - train_start_time,
                                                                                     train_loss, test_loss))
            ftest.write('{}\n'.format(test_loss))

        mse = test(model, test_dataloader, fold + 1, 'Test', True)
        test(model, train_dataloader, fold + 1, 'Training', True)
        sum_mse = sum_mse + mse
        print('FOLD {} OF {} COMPLETED!'.format(fold + 1, split_fold))
        print('*********************************************************')
        ftrain.write('*\n')
        ftest.write('*\n')
    print('')
    print('TRAINING ENDED!')
    print('Average MSE across {} folds is {}.'.format(split_fold, sum_mse / split_fold))
    ftrain.close()
    ftest.close()

    # print('*********************************************************')
    # test(model, full_training, -1, True)

