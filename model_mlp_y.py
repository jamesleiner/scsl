import pandas as pd
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import argparse
import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn
import pickle
import multiprocessing as mp 
import warnings
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from hrt_funs import *
import math
from scipy.stats import norm
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
import time
warnings.filterwarnings("ignore")

datadir = 'data/'
modeldir = 'models/'


class BigNet(nn.Module):
    def __init__(self, n_inputs):
        super(BigNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        # self.m = {}
        # self.update_masks() # builds the initial self.m connectivity
    
    def forward(self, x):
        #Input to the first hidden layer
        x = self.fc1(x)
        x=self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # Third hidden layer
        return torch.sigmoid(x)

def create_masked_data_X(X,Y,X_target,prob_mask = 0.2):
    yinds = torch.distributions.bernoulli.Bernoulli(probs=torch.tensor([prob_mask])).sample([Y.shape[1]])
    yinds = torch.nonzero(yinds.view(-1)) 
    ymask = torch.ones(Y.shape[1])
    ymask[yinds] = 0
    Y_nn = (2*Y-1)*ymask
    
    
    X_nn = torch.cat((X[:,:X_target],X[:,(X_target+1):]),1)
    return(torch.cat((X_nn,Y_nn == -1, Y_nn==1), 1).float())


def create_masked_data_Y(X,Y,Y_target,prob_mask = 0.2):
    yinds = torch.distributions.bernoulli.Bernoulli(probs=torch.tensor([prob_mask])).sample([Y.shape[1]])
    yinds = torch.nonzero(yinds.view(-1)) 
    ymask = torch.ones(Y.shape[1])
    ymask[yinds] = 0
    ymask[Y_target] = 0
    Y_nn = (2*Y-1)*ymask
    
    xind = torch.distributions.categorical.Categorical(probs=torch.tensor(torch.ones([X.shape[1]])/X.shape[1])).sample([1])
    xmask = torch.ones(X.shape[1])
    xmask[xind] = 0
    X_nn = (2*X-1)*xmask
    return(torch.cat((X_nn==-1,X_nn==1,Y_nn == -1, Y_nn==1), 1).float())


# Parse command line arguments
# X_target - Index of target feature in the X dataset
# label - name of dataset
# NUM_EPOCHS - number of epochs to train
# BATCH_SIZE - batch size for training neural net
# PROB_MASK - probability of masking variable during model training
# LR - learning rate
# num_X_mask - index of X values below which will be excluded (default -1 indicates all features considered)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--Y_target',type=int,default =0)
parser.add_argument('--label',type=str,default = "synth")
parser.add_argument('--NUM_EPOCHS',type=int,default =200)
parser.add_argument('--BATCH_SIZE',type=int,default =50)
parser.add_argument('--PROB_MASK',type=float,default =0.2)
parser.add_argument('--LR',type=float,default =1e-2)
parser.add_argument('--num_X_mask',type=int,default =-1)
args = parser.parse_args()

label = args.label
Y_target= args.Y_target
NUM_EPOCHS = args.NUM_EPOCHS
BATCH_SIZE = args.BATCH_SIZE
PROB_MASK = args.PROB_MASK
LR = args.LR
num_X_mask = args.num_X_mask



#create masked data
def create_masked_data_Y(X,Y,Y_target,prob_mask = 0.2,num_X_mask =47):
    yinds = torch.distributions.bernoulli.Bernoulli(probs=torch.tensor([prob_mask])).sample([Y.shape[1]])
    yinds = torch.nonzero(yinds.view(-1)) 
    ymask = torch.ones(Y.shape[1])
    ymask[yinds] = 0
    ymask[Y_target] = 0
    Y_nn = (2*Y-1)*ymask
    
    if num_X_mask < 1:
        xind = torch.distributions.categorical.Categorical(probs=torch.tensor(torch.ones([X.shape[1]])/X.shape[1])).sample([1])
        xmask = torch.ones(X.shape[1])
        xmask[xind] = 0
        X_nn = (2*X-1)*xmask
        feats = torch.cat((X_nn==-1,X_nn==1,Y_nn == -1, Y_nn==1), 1).float()
    else:
        X_genes = X[:,:num_X_mask]
        X_primary_site = X[:,num_X_mask:]
        xind = torch.distributions.categorical.Categorical(probs=torch.tensor(torch.ones([num_X_mask])/num_X_mask)).sample([1])
        xmask = torch.ones(num_X_mask)
        xmask[xind] = 0
        X_nn = (2*X_genes-1)*xmask
        feats = torch.cat((X_nn==-1,X_nn==1,X_primary_site.bool(),Y_nn == -1, Y_nn==1), 1).float()

    return feats

#train model
def train_model(model,X,Y,target,NUM_EPOCHS = 200, BATCH_SIZE = 50, PROB_MASK = 0.2, LR = 1e-2,filesave="model.pkl",type="X",num_X_mask = -1):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    train_losses = []
    for epoch in trange(NUM_EPOCHS, desc="train epochs"):
        print(epoch)
        model.train()
        this_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            if(type == "X"):
                inputs = create_masked_data_X(batch_x,batch_y,target,prob_mask = PROB_MASK)
            else:
                inputs = create_masked_data_Y(batch_x,batch_y,target,prob_mask = PROB_MASK,num_X_mask = num_X_mask)
            logits = model(inputs)

            # calculate loss.
            if(type == "X"):
                loss = criterion(logits.view(-1), batch_x[:,target].float())
            else:
                loss = criterion(logits.view(-1), batch_y[:,target].float())
            # autograd backward pass (calculates derivatives using backprop).
            loss.backward()
            # take SGD step (updates model weights).
            optimizer.step()
            this_losses.append(float(loss))
        train_losses.append(np.mean(this_losses))
    return train_losses


x_ds = label +'_x.npy'
y_ds = label + '_y.npy'
X = torch.from_numpy(np.load(datadir + x_ds))
Y = torch.from_numpy(np.load(datadir + y_ds))


train_dataset = TensorDataset(X,Y)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
criterion = torch.nn.BCELoss()
if(num_X_mask > 1):
    model_y = BigNet(2*num_X_mask + (X.shape[1]- num_X_mask)+2*Y.shape[1])
else:
    model_y = BigNet(2*X.shape[1]+2*Y.shape[1])
    
train_model(model_y,X,Y,Y_target,type="Y",NUM_EPOCHS = NUM_EPOCHS,BATCH_SIZE = 50, PROB_MASK = 0.2, LR = 1e-2, num_X_mask = num_X_mask)
filesave = modeldir +"Predictive Models/" +label + "_torchmodel" + "_ymodel_" + str(Y_target) + ".pkl"
print(filesave)
torch.save(model_y, filesave)
yo = torch.load(filesave)