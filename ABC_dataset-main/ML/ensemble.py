import torch

from scipy import stats 

import networkx as nx
import numpy as np

from torch_geometric.data import Dataset, download_url
from torch_geometric.data import DataLoader

#-------------------Note---------------------------------------------------------#
# This code is used to perform hard and soft voting
#-------------------Load all Segmentation Dataset--------------------------------#
from simple_norm_segment_pyg_graph import *  
#------------------Get validation dataset labels---------------------------------#
dataset_og_sparse = simple_graph_norm_segsparse(root = 'Graphs/sparse/normalized/') # original dataset


num_train = int(0.80*len(dataset_og_sparse)) # 80% of total data is used for training
num_val = int((len(dataset_og_sparse)-int(0.80*len(dataset_og_sparse)))/2)

print('Number of training data points:',num_train)
print('Number of validation dataset:',num_val)
torch.manual_seed(12345)
index = torch.randperm(25000)
dataset_og_sparse = dataset_og_sparse[index]

val_dataset_sparse = dataset_og_sparse[num_train:num_train+num_val]
test_dataset_sparse = dataset_og_sparse[num_train+num_val:]

labels = np.zeros(2500)


for ii in np.arange(0,len(labels)):
  labels[ii] = test_dataset_sparse[ii].y

#-------------------Load Hard Predictions------------------------------------#
testsparse_ball20_hard = np.load('test_results/raw_predictions/seg/class/testsparse_ball20.npy')
testsparse_ball30_hard = np.load('test_results/raw_predictions/seg/class/testsparse_ball30.npy')
testsparse_ball40_hard = np.load('test_results/raw_predictions/seg/class/testsparse_ball40.npy')

testmedium_ball20_hard = np.load('test_results/raw_predictions/seg/class/testmedium_ball20.npy')
testmedium_ball30_hard = np.load('test_results/raw_predictions/seg/class/testmedium_ball30.npy')
testmedium_ball40_hard = np.load('test_results/raw_predictions/seg/class/testmedium_ball40.npy')

testdense_ball20_hard = np.load('test_results/raw_predictions/seg/class/testdense_ball20.npy')
testdense_ball30_hard = np.load('test_results/raw_predictions/seg/class/testdense_ball30.npy')
testdense_ball40_hard = np.load('test_results/raw_predictions/seg/class/testdense_ball40.npy')

#-------------------Load Soft Predictions------------------------------------#
testsparse_ball20_soft = np.load('test_results/raw_predictions/seg/prob/testsparse_ball20.npy')
testsparse_ball30_soft = np.load('test_results/raw_predictions/seg/prob/testsparse_ball30.npy')
testsparse_ball40_soft = np.load('test_results/raw_predictions/seg/prob/testsparse_ball40.npy')

testmedium_ball20_soft = np.load('test_results/raw_predictions/seg/prob/testmedium_ball20.npy')
testmedium_ball30_soft = np.load('test_results/raw_predictions/seg/prob/testmedium_ball30.npy')
testmedium_ball40_soft = np.load('test_results/raw_predictions/seg/prob/testmedium_ball40.npy')

testdense_ball20_soft = np.load('test_results/raw_predictions/seg/prob/testdense_ball20.npy')
testdense_ball30_soft = np.load('test_results/raw_predictions/seg/prob/testdense_ball30.npy')
testdense_ball40_soft = np.load('test_results/raw_predictions/seg/prob/testdense_ball40.npy')

#-------------------Process Hard Vote--------------------------#
votesparse_ball20_hard,_ = stats.mode(testsparse_ball20_hard, axis = 0)
votesparse_ball30_hard,_ = stats.mode(testsparse_ball30_hard, axis = 0)
votesparse_ball40_hard,_ = stats.mode(testsparse_ball40_hard, axis = 0)

votemedium_ball20_hard,_ = stats.mode(testmedium_ball20_hard, axis = 0)
votemedium_ball30_hard,_ = stats.mode(testmedium_ball30_hard, axis = 0)
votemedium_ball40_hard,_ = stats.mode(testmedium_ball40_hard, axis = 0)

votedense_ball20_hard,_ = stats.mode(testdense_ball20_hard, axis = 0)
votedense_ball30_hard,_ = stats.mode(testdense_ball30_hard, axis = 0)
votedense_ball40_hard,_ = stats.mode(testdense_ball40_hard, axis = 0)

accsparse_ball20_hard = np.mean(votesparse_ball20_hard==labels)
accsparse_ball30_hard = np.mean(votesparse_ball30_hard==labels)
accsparse_ball40_hard = np.mean(votesparse_ball40_hard==labels)

accmedium_ball20_hard = np.mean(votemedium_ball20_hard==labels)
accmedium_ball30_hard = np.mean(votemedium_ball30_hard==labels)
accmedium_ball40_hard = np.mean(votemedium_ball40_hard==labels)

accdense_ball20_hard = np.mean(votedense_ball20_hard==labels)
accdense_ball30_hard = np.mean(votedense_ball30_hard==labels)
accdense_ball40_hard = np.mean(votedense_ball40_hard==labels)

print('Hard Voting:')
print('Sparse r = 20: ', accsparse_ball20_hard)
print('Sparse r = 30: ', accsparse_ball30_hard)
print('Sparse r = 40: ', accsparse_ball40_hard)

print('Medium r = 20: ', accmedium_ball20_hard)
print('Medium r = 30: ', accmedium_ball30_hard)
print('Medium r = 40: ', accmedium_ball40_hard)

print('Dense r = 20: ', accdense_ball20_hard)
print('Dense r = 30: ', accdense_ball30_hard)
print('Dense r = 40: ', accdense_ball40_hard)

np.save('test_results/voting/seg/hard_predictions.npy', 
        np.asarray([accsparse_ball20_hard,accsparse_ball30_hard,accsparse_ball40_hard,
        accmedium_ball20_hard,accmedium_ball30_hard,accmedium_ball40_hard,
        accdense_ball20_hard,accdense_ball30_hard,accdense_ball40_hard])) # Save hard voting results

#------------------------------Process Soft Vote--------------------------------#
probsparse_ball20_soft = np.mean(testsparse_ball20_soft,axis = 0)
probsparse_ball30_soft = np.mean(testsparse_ball30_soft,axis = 0)
probsparse_ball40_soft = np.mean(testsparse_ball40_soft,axis = 0)

probmedium_ball20_soft = np.mean(testmedium_ball20_soft,axis = 0)
probmedium_ball30_soft = np.mean(testmedium_ball30_soft,axis = 0)
probmedium_ball40_soft = np.mean(testmedium_ball40_soft,axis = 0)

probdense_ball20_soft = np.mean(testdense_ball20_soft,axis = 0)
probdense_ball30_soft = np.mean(testdense_ball30_soft,axis = 0)
probdense_ball40_soft = np.mean(testdense_ball40_soft,axis = 0)

votesparse_ball20_soft = np.argmax(probsparse_ball20_soft, axis = 1)
votesparse_ball30_soft = np.argmax(probsparse_ball30_soft, axis = 1)
votesparse_ball40_soft = np.argmax(probsparse_ball40_soft, axis = 1)

votemedium_ball20_soft = np.argmax(probmedium_ball20_soft, axis = 1)
votemedium_ball30_soft = np.argmax(probmedium_ball30_soft, axis = 1)
votemedium_ball40_soft = np.argmax(probmedium_ball40_soft, axis = 1)

votedense_ball20_soft = np.argmax(probdense_ball20_soft, axis = 1)
votedense_ball30_soft = np.argmax(probdense_ball30_soft, axis = 1)
votedense_ball40_soft = np.argmax(probdense_ball40_soft, axis = 1)

accsparse_ball20_soft = np.mean(votesparse_ball20_soft==labels)
accsparse_ball30_soft = np.mean(votesparse_ball30_soft==labels)
accsparse_ball40_soft = np.mean(votesparse_ball40_soft==labels)

accmedium_ball20_soft = np.mean(votemedium_ball20_soft==labels)
accmedium_ball30_soft = np.mean(votemedium_ball30_soft==labels)
accmedium_ball40_soft = np.mean(votemedium_ball40_soft==labels)

accdense_ball20_soft = np.mean(votedense_ball20_soft==labels)
accdense_ball30_soft = np.mean(votedense_ball30_soft==labels)
accdense_ball40_soft = np.mean(votedense_ball40_soft==labels)

print('Soft Voting:')
print('Sparse r = 20: ', accsparse_ball20_soft)
print('Sparse r = 30: ', accsparse_ball30_soft)
print('Sparse r = 40: ', accsparse_ball40_soft)

print('Medium r = 20: ', accmedium_ball20_soft)
print('Medium r = 30: ', accmedium_ball30_soft)
print('Medium r = 40: ', accmedium_ball40_soft)

print('Dense r = 20: ', accdense_ball20_soft)
print('Dense r = 30: ', accdense_ball30_soft)
print('Dense r = 40: ', accdense_ball40_soft)

np.save('test_results/voting/seg/soft_predictions.npy', 
        np.asarray([accsparse_ball20_soft,accsparse_ball30_soft,accsparse_ball40_soft,
        accmedium_ball20_soft,accmedium_ball30_soft,accmedium_ball40_soft,
        accdense_ball20_soft,accdense_ball30_soft,accdense_ball40_soft])) # Save soft voting results
