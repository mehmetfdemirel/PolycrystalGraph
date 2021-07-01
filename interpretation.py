# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:21:00 2021

@author: daimi
"""
import numpy as np
from numpy import savetxt
import torch
import argparse
from torch.autograd import Variable
import math
from interpretation_data import *
from model import GraphModel, Message_Passing
def tensor_to_variable_grad(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float(),requires_grad=True)

def variable_to_numpy_grad(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x

# calculate the gradient of points
def gradient_calculation(adjacency_matrix, node_attr_matrix, t_matrix, model):
    adjacency_matrix=tensor_to_variable_grad(adjacency_matrix)
    node_attr_matrix=tensor_to_variable_grad(node_attr_matrix)
    t_matrix=tensor_to_variable_grad(t_matrix)
    
    label=model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix, t_matrix=t_matrix)
    label.backward()  # back propagation
    grad_node_attr_matrix=variable_to_numpy_grad(node_attr_matrix.grad) # get the gradient of features
    grad_t_matrix=variable_to_numpy_grad(t_matrix.grad)  # get the gradient of external field
    
    return grad_node_attr_matrix, grad_t_matrix
    
def Intergrated_gradient_calculation(adajacency_matrix, node_attr_matrix, t_matrix, model,steps=200):
    nsize=np.asarray(node_attr_matrix.size())
    tsize=np.asarray(t_matrix.size())
    baseline_node_attr_matrix=torch.zeros((1,nsize[0],nsize[1]))
    baseline_t_matrix=torch.zeros((1,tsize[0]))

    ##specify baseline: same with input graph except the Euler angles
    ##specify baseline: all zeros
    #specify baseline: only consider the influence of grain size
    #alpha = 0.5*math.pi
    #beta = 0.5*math.pi
    #gamma = 0.5*math.pi
    for i in range(nsize[0]):
    #    if node_attr_matrix[i][0] != 0:
    #        baseline_node_attr_matrix[0][i][0]=alpha
    #    if node_attr_matrix[i][1] != 0:
    #        baseline_node_attr_matrix[0][i][1]=beta
    #    if node_attr_matrix[i][2] != 0:
    #        baseline_node_attr_matrix[0][i][2]=gamma
        baseline_node_attr_matrix[0][i][0]=node_attr_matrix[i][0]
        baseline_node_attr_matrix[0][i][1]=node_attr_matrix[i][1]
        baseline_node_attr_matrix[0][i][2]=node_attr_matrix[i][2]
        baseline_node_attr_matrix[0][i][4]=node_attr_matrix[i][4]

    for i in range(tsize[0]):
        baseline_t_matrix[0][i]=t_matrix[i]
        
    adajacency_matrix=torch.reshape(adajacency_matrix,(1,nsize[0],nsize[0]))

    grad_node_attr_matrix=np.zeros((1,nsize[0],nsize[1]))
    grad_t_matrix=np.zeros((1,tsize[0]))
    
    for step in range(steps):
        temp_node_attr_matrix=np.zeros((1,nsize[0],nsize[1]))
        temp_t_matrix=np.zeros((1,tsize[0]))
        temp_node_attr_matrix=torch.from_numpy(temp_node_attr_matrix)
        temp_t_matrix=torch.from_numpy(temp_t_matrix)
        
        for i in range(nsize[0]):
            for j in range(nsize[1]):
                temp_node_attr_matrix[0][i][j]=baseline_node_attr_matrix[0][i][j]+(node_attr_matrix[i][j]-baseline_node_attr_matrix[0][i][j])*step/steps
                
        for i in range(tsize[0]):
            temp_t_matrix[0][i]=baseline_t_matrix[0][i]+(t_matrix[i]-baseline_t_matrix[0][i])*step/steps


        temp_grad_node_attr_matrix, temp_grad_t_matrix=gradient_calculation(adajacency_matrix, temp_node_attr_matrix, temp_t_matrix, model)
        grad_node_attr_matrix=grad_node_attr_matrix+temp_grad_node_attr_matrix
        grad_t_matrix=grad_t_matrix+temp_grad_t_matrix

    node_attr_matrix=node_attr_matrix.numpy()
    baseline_node_attr_matrix=baseline_node_attr_matrix.numpy()
    t_matrix=t_matrix.numpy()
    baseline_t_matrix=baseline_t_matrix.numpy()
        
    for i in range(nsize[0]):
            for j in range(nsize[1]):
                grad_node_attr_matrix[0][i][j]=(node_attr_matrix[i][j]-baseline_node_attr_matrix[0][i][j])*grad_node_attr_matrix[0][i][j]/steps
                
    for i in range(tsize[0]):
        grad_t_matrix[0][i]=(t_matrix[i]-baseline_t_matrix[0][i])*grad_t_matrix[0][i]/steps

    return grad_node_attr_matrix, grad_t_matrix  

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/')
    given_args = parser.parse_args()
    checkpoint_dir = given_args.checkpoint
    
    dataset = GraphDataSet_interpretable()
    model = torch.load('{}/checkpoint.pth'.format(checkpoint_dir))#,map_location=torch.device('cpu'))
    
    for i in range(492):
        Graph = dataset[i]
        adajacency_matrix, node_attr_matrix, t_matrix, label_matrix = Graph[0], Graph[1], Graph[2], Graph[3]
        grad_node_attr_matrix, grad_t_matrix=Intergrated_gradient_calculation(adajacency_matrix, node_attr_matrix, t_matrix, model,steps=200)
        feature_gradient=grad_node_attr_matrix[0]
        outputnumber=i+1
        savetxt("interpretation/feature_grad_{0}.csv".format(outputnumber), feature_gradient, delimiter=',')
        savetxt("interpretation/label_{0}.csv".format(outputnumber), label_matrix, delimiter=',')
        savetxt("interpretation/feature_{0}.csv".format(outputnumber),node_attr_matrix,delimiter=',')
    
