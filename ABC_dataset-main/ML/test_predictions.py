import torch
import torch.nn.functional as F
import numpy as np
import os.path
from pathlib import Path
import sys  

#-------------------Note--------------------------------------------#
# This code is used to get soft and hard voting predictions
#-------------------Load all Segmentation Dataset-------------------#
from pyg_graphs import * 

#-----------------------Load Model-----------------------------------#
from Pointnet_layer import * 

nums_data = np.asarray([20])#np.asarray([5,10,15,20])
seeds = np.arange(1,11,dtype=int)
radius = np.asarray([20,30,40])#np.asarray([20,30,40])


#--------------------File path for saved states-----------------------------#
f_sparse = 'ML_results/ball/seg/4_layers_1MLP_skip_emb64_AdamW_aug/sparse/'
f_medium = 'ML_results/ball/seg/4_layers_1MLP_skip_emb64_AdamW_aug/medium/'
f_dense = 'ML_results/ball/seg/4_layers_1MLP_skip_emb64_AdamW_aug/dense/'

#----------------------Use GPU if avialable---------------------------------#
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(DEVICE)

#----------------------Load models---------------------------------------#
model_sparse = PointNet4Layers()
model_sparse.to(DEVICE)
model_medium = PointNet4Layers()
model_medium.to(DEVICE)
model_dense = PointNet4Layers()
model_dense.to(DEVICE)

#-----------------------Split Models into training/test/validation--------------#
dataset_og_sparse = simple_graph_norm_segsparse(root = 'Graphs/sparse/normalized/') # original dataset
dataset_og_medium = simple_graph_norm_segmedium(root = 'Graphs/medium/normalized/') # original dataset
dataset_og_dense = simple_graph_norm_segdense(root = 'Graphs/dense/normalized/') #original dataset

num_train = int(0.80*len(dataset_og_sparse)) # 80% of total data is used for training
num_val = int((len(dataset_og_sparse)-int(0.80*len(dataset_og_sparse)))/2)

print('Number of training data points:',num_train)
print('Number of validation dataset:',num_val)
torch.manual_seed(12345)
index = torch.randperm(25000)
dataset_og_sparse = dataset_og_sparse[index]
dataset_og_medium = dataset_og_medium[index]
dataset_og_dense = dataset_og_dense[index]

#---------------------Dataloaders-------------------------------------------#
val_dataset_sparse = dataset_og_sparse[num_train:num_train+num_val]
val_dataset_medium = dataset_og_medium[num_train:num_train+num_val]
val_dataset_dense = dataset_og_dense[num_train:num_train+num_val]

test_dataset_sparse = dataset_og_sparse[num_train+num_val:]
test_dataset_medium = dataset_og_medium[num_train+num_val:]
test_dataset_dense = dataset_og_dense[num_train+num_val:]

val_loader_sparse = DataLoader(val_dataset_sparse, batch_size = 64, shuffle = False,num_workers = 1, pin_memory=True)
val_loader_medium = DataLoader(val_dataset_medium, batch_size = 64, shuffle = False,num_workers = 1, pin_memory=True)
val_loader_dense = DataLoader(val_dataset_dense, batch_size = 64, shuffle = False,num_workers = 1, pin_memory=True)

test_loader_sparse = DataLoader(test_dataset_sparse, batch_size = 64, shuffle = False,num_workers = 1, pin_memory=True)
test_loader_medium = DataLoader(test_dataset_medium, batch_size = 64, shuffle = False,num_workers = 1, pin_memory=True)
test_loader_dense = DataLoader(test_dataset_dense, batch_size = 64, shuffle = False,num_workers = 1, pin_memory=True)

@torch.no_grad()

#---------------Test function--------------------------------#
def test(model, loader,r):
    
    model.eval()

    pred_all = []
    prob_all = []
    
    for data in loader:
        data = data.to(DEVICE)
        logits = model(data.pos,data.x,data.batch,r)
        prob = F.softmax(logits, dim=1)
        prob_all.append(prob.detach().cpu().numpy())
        pred = torch.argmax(logits,dim=1)
        pred_all.append(pred.detach().cpu().numpy().reshape(-1))
    pred_all = np.concatenate(pred_all, axis=0)
    prob_all = np.concatenate(prob_all, axis=0)
    return pred_all, prob_all # outputs class (for hard voting) and probability distribution (for soft voting )


for r in radius:
  print('Radius:',r)
  test_sparse_pred_all = []
  test_medium_pred_all = []
  test_dense_pred_all = []
  
  test_sparse_prob_all = []
  test_medium_prob_all = []
  test_dense_prob_all = []
  

  for seed in seeds:
    for num_data in nums_data:
      #-------------------Load Model Weights--------------------------------------------------------#
      check_sparse = torch.load(f_sparse+str(num_data)+'kpoints/ball'+str(r)+'/seed'+str(seed)+'/model_state/epoch50.pt', map_location = torch.device(DEVICE))
      check_medium = torch.load(f_medium+str(num_data)+'kpoints/ball'+str(r)+'/seed'+str(seed)+'/model_state/epoch50.pt', map_location = torch.device(DEVICE))
      check_dense = torch.load(f_dense+str(num_data)+'kpoints/ball'+str(r)+'/seed'+str(seed)+'/model_state/epoch50.pt', map_location = torch.device(DEVICE))
      
      #-------------------Load Model-----------------------------------------------------------------#
      model_sparse.load_state_dict(check_sparse['model_state_dict'])
      model_medium.load_state_dict(check_medium['model_state_dict'])
      model_dense.load_state_dict(check_dense['model_state_dict'])
      
      #---------------------Test Model----------------------------------------------------------------#
      test_sparse_pred, test_sparse_prob = test(model_sparse,test_loader_sparse,r)
      test_medium_pred, test_medium_prob = test(model_medium,test_loader_medium,r)
      test_dense_pred, test_dense_prob = test(model_dense,test_loader_dense,r)
  
  
    test_sparse_pred_all.append(test_sparse_pred)
    test_sparse_prob_all.append(test_sparse_prob)
    
    test_medium_pred_all.append(test_medium_pred)
    test_medium_prob_all.append(test_medium_prob)
    
    test_dense_pred_all.append(test_dense_pred)
    test_dense_prob_all.append(test_dense_prob)
  
  test_sparse_pred_all = np.asarray(test_sparse_pred_all)
  test_sparse_prob_all = np.asarray(test_sparse_prob_all)
  
  test_medium_pred_all = np.asarray(test_medium_pred_all)
  test_medium_prob_all = np.asarray(test_medium_prob_all)

  test_dense_pred_all = np.asarray(test_dense_pred_all)
  test_dense_prob_all = np.asarray(test_dense_prob_all)


#---------------------Save predictions--------------------------------------------# 
  np.save('test_results/raw_predictions/seg/class/test_sparse_ball'+str(r)+'.npy',np.asarray(test_sparse_pred_all)) # for hard voting
  np.save('test_results/raw_predictions/seg/prob/test_sparse_ball'+str(r)+'.npy',np.asarray(test_sparse_prob_all)) # for soft voting
  
  np.save('test_results/raw_predictions/seg/class/test_medium_ball'+str(r)+'.npy',np.asarray(test_medium_pred_all)) # for hard voting
  np.save('test_results/raw_predictions/seg/prob/test_medium_ball'+str(r)+'.npy',np.asarray(test_medium_prob_all)) # for soft voting
  
  np.save('test_results/raw_predictions/seg/class/test_dense_ball'+str(r)+'.npy',np.asarray(test_dense_pred_all)) # for hard voting
  np.save('test_results/raw_predictions/seg/prob/test_dense_ball'+str(r)+'.npy',np.asarray(test_dense_prob_all)) # for soft voting
       

  