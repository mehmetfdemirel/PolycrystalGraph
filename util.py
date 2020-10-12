import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from train_data import GraphDataSet

def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())

def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x

def print_preds(y_label_list, y_pred_list, test_or_tr):
    length, w = np.shape(y_label_list)
    print()
    print('{} Set Predictions: '.format(test_or_tr))
    for i in range(0, length):
        print('True:{}, Predicted: {}'.format(y_label_list[i], y_pred_list[i]))

def get_data(idx_path, running_index, folds, batch_size):
    indices = np.load(idx_path, allow_pickle=True)['indices']
    test_idx = indices[running_index]
    train_idx = indices[[i for i in range(folds) if i != running_index]]
    train_idx = [item for sublist in train_idx for item in sublist]

    dataset = GraphDataSet()
    train_data = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    test_data = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))
    return train_data, test_data

def mse(Y_prime, Y):
    return np.mean((Y_prime - Y) ** 2)

def macro_avg_err(Y_prime, Y):
    if type(Y_prime) is np.ndarray:
        return np.abs(np.sum((Y - Y_prime)) / np.sum(Y))
    return torch.abs(torch.sum(Y - Y_prime) / torch.sum(Y))
