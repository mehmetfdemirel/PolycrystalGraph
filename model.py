from __future__ import print_function

import argparse
import time
from collections import OrderedDict
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from util import *
from data import *

class Message_Passing(nn.Module):
    def forward(self, x, adjacency_matrix):
        neighbor_nodes = torch.bmm(adjacency_matrix, x)
        logging.debug('neighbor message\t', neighbor_nodes.size())
        logging.debug('x shape\t', x.size())
        return x

class GraphModel(nn.Module):
    def __init__(self, max_node_num, atom_attr_dim, latent_dim1, latent_dim2):
        super(GraphModel, self).__init__()

        self.max_node_num = max_node_num
        self.atom_attr_dim = atom_attr_dim
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2

        self.graph_modules = nn.Sequential(OrderedDict([
            ('message_passing_0', Message_Passing()),
            ('dense_0', nn.Linear(self.atom_attr_dim, self.latent_dim2)),
            ('activation_0', nn.Sigmoid()),
            ('message_passing_1', Message_Passing()),
            ('dense_1', nn.Linear(self.latent_dim2, self.latent_dim1)),
            ('activation_1', nn.Sigmoid()),
        ]))

        self.fully_connected = nn.Sequential(
            nn.Linear(self.max_node_num * self.latent_dim1 + 1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
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

def train(model, train_data_loader, validation_data_loader, epochs, checkpoint_dir, optimizer, criterion, validation_index, folder_name):
    print()
    print("*** Training started! ***")
    print()
    
    filename='{}/learning_Output_{}.txt'.format(folder_name, validation_index)
    output=open(filename, "w")
    print('Epoch Training_time Training_MSE Validation_MSE',file=output, flush = True)  

    for epoch in range(epochs):
        model.train()
        total_macro_loss = []
        total_mse_loss = []
      #  if epoch % (epochs / 10) == 0 or epoch == epochs-1:
      #      torch.save(model.state_dict(), '{}/checkpoint_{}.pth'.format(checkpoint_dir, epoch))
      #      print('Epoch: {}, Checkpoint saved!'.format(epoch))
      #  else:
      #      print('Epoch: {}'.format(epoch))

        train_start_time = time.time()

        for batch_id, (adjacency_matrix, node_attr_matrix, t_matrix, label_matrix) in enumerate(train_data_loader):
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

        train_end_time = time.time()
        _, training_loss_epoch = test(model, train_data_loader, 'Training', False, criterion, validation_index, folder_name) 
        _, validation_loss_epoch = test(model, validation_dataloader, 'Validation', False, criterion, validation_index, folder_name)
        print('%d %.3f %e %e' % (epoch, train_end_time-train_start_time, training_loss_epoch, validation_loss_epoch), file=output,flush=True )

def test(model, data_loader, test_val_tr, printcond, criterion, running_index, folder_name):
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

    norm = np.load('norm.npz', allow_pickle=True)['norm']
    label_mean, label_std = norm[0], norm[1]

    y_label_list = np.array(y_label_list) * label_std + label_mean
    y_pred_list = np.array(y_pred_list) * label_std + label_mean

    total_loss = macro_avg_err(y_pred_list, y_label_list)
    total_mse = criterion(torch.from_numpy(y_pred_list), torch.from_numpy(y_label_list)).item()

    length, w = np.shape(y_label_list)
    if printcond:
        filename = '{}/{}_Output_{}.txt'.format(folder_name, test_val_tr, running_index)
        output = open(filename, 'w')
        #print()
        print('{} Set Predictions: '.format(test_val_tr), file = output, flush = True)
        print('True_value Predicted_value', file=output, flush = True)
        for i in range(0, length):
            print('%f, %f' % (y_label_list[i], y_pred_list[i]),file=output,flush = True)

    return total_loss, total_mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_node_num', type=int, default=300)
    parser.add_argument('--atom_attr_dim', type=int, default=5)
    parser.add_argument('--num_graphs', type=int, default=492)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--min_learning_rate', type=float, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/')
    parser.add_argument('--validation_index', type=int, default=0)
    parser.add_argument('--testing_index', type=int, default=1)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--idx_path', type=str, default='indices_and_graphseq.npz')
    parser.add_argument('--folder_name', type=str, default='output/')
    parser.add_argument('--num_data', type=int, default=492)
    parser.add_argument('--hyper',type=int,default=0)

    given_args = parser.parse_args()
    max_node_num = given_args.max_node_num
    atom_attr_dim = given_args.atom_attr_dim
    num_graphs = given_args.num_graphs
    checkpoint_dir = given_args.checkpoint
    validation_index = given_args.validation_index
    testing_index = given_args.testing_index
    idx_path = given_args.idx_path
    folds = given_args.folds
    batch_size = given_args.batch_size
    min_learning_rate = given_args.min_learning_rate
    seed = given_args.seed
    checkpoint_dir = given_args.checkpoint
    folds = given_args.folds
    idx_path = given_args.idx_path
    folder_name = given_args.folder_name
    num_data = given_args.num_data
    hyper=given_args.hyper

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    os.environ['PYTHONHASHargs.seed'] = str(given_args.seed)
    np.random.seed(given_args.seed)
    torch.manual_seed(given_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(given_args.seed)
        torch.cuda.manual_seed_all(given_args.seed)
    torch.backends.cudnn.deterministic = True
    
    filename='hyper/'+str(hyper)+'.json'
    with open(filename,'r') as h:
        hyperset=json.load(h)
        
    latent_dim1=hyperset['latent_dim1']
    latent_dim2=hyperset['latent_dim2']
    epochs=hyperset['epoch']
    learning_rate=hyperset['lr']
    in_optim=hyperset['optim']

    # Define the model
    model = GraphModel(max_node_num, atom_attr_dim, latent_dim1, latent_dim2)
    if torch.cuda.is_available():
        model.cuda()
        
    if in_optim=="Adam":
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    elif in_optim=="RMSprop":
        optimizer = optim.RMSprop(model.parameters(),lr=learning_rate)
    elif in_optim=="SGD":
        optimizer = optim.SGD(model.parameters(),lr=learning_rate)
        
    criterion = nn.MSELoss()

    # get the data
    train_dataloader, validation_dataloader, test_dataloader = get_data(batch_size, idx_path, validation_index, testing_index, folds, num_data)

    # train the model
    train_start_time = time.time()
    train(model, train_dataloader, validation_dataloader,epochs, checkpoint_dir, optimizer, criterion, validation_index,folder_name)
    train_end_time = time.time()

    torch.save(model, '{}/checkpoint.pth'.format(checkpoint_dir))    
    
    # predictions on the entire training and test datasets
    train_rel, train_mse= test(model, train_dataloader, 'Training', True, criterion, validation_index, folder_name)
    validation_rel, validation_mse=test(model, validation_dataloader, 'Validation', True, criterion, validation_index, folder_name)
    test_start_time = time.time()
    test_rel, test_mse= test(model, test_dataloader, 'Test', True, criterion, testing_index, folder_name)
    test_end_time = time.time()

    
    print('--------------------')
    print("validation_index : {}".format(validation_index))
    print("testing_index : {}".format(testing_index))
    print("training_time : {}".format(train_end_time-train_start_time))
    print("testing_time : {}".format(test_end_time-test_start_time))
    print("Train Relative Error: {:.3f}%".format(100 * train_rel))
    print("Validation Relative Error: {:.3f}%".format(100 * validation_rel))
    print("Test Relative Error: {:.3f}%".format(100 * test_rel))
    print("Train MSE : {}".format(train_mse))
    print("Validation MSE : {}".format(validation_mse))
    print("Test MSE: {}".format(test_mse))



