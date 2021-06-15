import torch
import numpy as np
from torch.autograd import Variable


def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())
    #return Variable(x)

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

def mse(Y_prime, Y):
    return np.mean((Y_prime - Y) ** 2)

def macro_avg_err(Y_prime, Y):
    if type(Y_prime) is np.ndarray:
        return np.sum(np.abs(Y - Y_prime)) / np.sum(np.abs(Y))
    return torch.sum(torch.abs(Y - Y_prime)) / torch.sum(torch.abs(Y))
