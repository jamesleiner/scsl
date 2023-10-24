'''
optimize p_vals: Given already trained models (either logistic or MLP), this script searches for the conditioning subset which will generate the highest p-value. 

Expected command line arguments:

- X_target: Column of X matrix to consider in testing for edge
- Y_target: Coluimn of Y matrix to consider in testing for edge
- NUM_EPOCHS: Number of epochs to search for highest p-value
- stop_pval: P-value to use for early stopping rule
- stop_train: Number of epochs to train model prior to using hybrid search approach
- model_type: Either mlp or logit
- label: Dataset name
- search_type: Will exhaustively search over all subsets if True (recommended False)
- start_X_primary: Only use if there are indicators to be included in all models. Default to -1
- subset_primary: Only use if there is a subsetting criterion to restrict the dataset to 
- learning_r: Learning rate to use in optimization
'''

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
import sklearn
import pickle
import multiprocessing as mp 
import warnings
import itertools
import math
import random
import time
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, latent_dim,categorical_dim,hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


def gcm(X_j,Y_k, pred_Xj, pred_Yk):
    R = (X_j -pred_Xj)*(pred_Yk - Y_k)
    n = R.shape[0]
    numerator = torch.sum(R)*math.sqrt(n)/n
    denom = math.sqrt(torch.mean(R**2) - (torch.mean(R)**2))
    T = abs(numerator/denom)
    return T


def get_weights(t,pr):
    prob_pick = np.prod((1-pr)[[x for x in possible_vars if x not in t]])* np.prod(pr[[list(t)]])
    return prob_pick

class BigNet(nn.Module):
    def __init__(self, n_inputs):
        super(BigNet,self).__init__()
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

class subset_optim(nn.Module):
    def __init__(self, default_prob,X_dim,Y_dim,model_x,model_y,X_target,Y_target,model_type = "mlp",search_type = "univariate",start_X_primary=-1):
        super(subset_optim, self).__init__()
        self.start_probs = np.array([[logit(default_prob),logit(1- default_prob)] for i in range(Y_dim)])
        self.relu = nn.ReLU()
        self.X_dim = X_dim
        self.Y_dim = Y_dim
        self.model_type = model_type
        self.search_type = search_type
        self.X_target = X_target
        self.Y_target = Y_target
        self.start_X_primary = start_X_primary
        
        if(self.search_type == "univariate"):
            self.logs = nn.Parameter(torch.FloatTensor(self.start_probs))
        else:
            self.bias = nn.Parameter(torch.FloatTensor(self.start_probs[:,0]))
            self.weight = nn.Parameter(torch.zeros(Y_dim-1,Y_dim))
        
        if(self.model_type == "mlp"):
            self.init_mlp()
        else:
            self.init_logit()
    
    def init_logit(self):
        self.coefs_X = torch.transpose(torch.from_numpy(model_x.coef_.astype(np.float32)),0,1)
        self.intercept_X = torch.from_numpy(model_x.intercept_.astype(np.float32))
        self.coefs_Y = torch.transpose(torch.from_numpy(model_y.coef_.astype(np.float32)),0,1)
        self.intercept_Y = torch.from_numpy(model_y.intercept_.astype(np.float32))
        
        
    def init_mlp(self):
        self.fc1_x = nn.Linear(X.shape[1]+2*Y.shape[1]-1,200)           
        self.fc2_x = nn.Linear(200, 200)
        self.fc3_x = nn.Linear(200, 1)
        self.fc1_x.weight = nn.Parameter(model_x.fc1.weight, requires_grad=False)
        self.fc1_x.bias = nn.Parameter(model_x.fc1.bias, requires_grad=False)
        self.fc2_x.weight = nn.Parameter(model_x.fc2.weight, requires_grad=False)
        self.fc2_x.bias = nn.Parameter(model_x.fc2.bias, requires_grad=False)
        self.fc3_x.weight = nn.Parameter(model_x.fc3.weight, requires_grad=False)
        self.fc3_x.bias = nn.Parameter(model_x.fc3.bias, requires_grad=False)
        
        if(self.start_X_primary > 1):
            self.fc1_y = nn.Linear(2*X.shape[1] - self.start_X_primary +2*Y.shape[1],1)
        else:
            self.fc1_y = nn.Linear(2*X.shape[1]+2*Y.shape[1],1)
            
        self.fc2_y = nn.Linear(200, 200)
        self.fc3_y = nn.Linear(200, 1)
        self.fc1_y.weight = nn.Parameter(model_y.fc1.weight, requires_grad=False)
        self.fc1_y.bias = nn.Parameter(model_y.fc1.bias, requires_grad=False)
        self.fc2_y.weight = nn.Parameter(model_y.fc2.weight, requires_grad=False)
        self.fc2_y.bias = nn.Parameter(model_y.fc2.bias, requires_grad=False)
        self.fc3_y.weight = nn.Parameter(model_y.fc3.weight, requires_grad=False)
        self.fc3_y.bias = nn.Parameter(model_y.fc3.bias, requires_grad=False)
    
    def get_pred_mlp_Y(self,x):
        x = self.fc1_y(x)
        x = self.relu(x)
        x = self.fc2_y(x)
        x = self.relu(x)
        x = self.fc3_y(x)
        return torch.sigmoid(x.view(-1))
    
    def get_pred_mlp_X(self,x):
        x = self.fc1_x(x)
        x = self.relu(x)
        x = self.fc2_x(x)
        x = self.relu(x)
        x = self.fc3_x(x)
        return torch.sigmoid(x.view(-1))
    
    def get_pred_logit_Y(self,x):
        logits = torch.matmul(x,self.coefs_Y) + self.intercept_Y
        return torch.sigmoid(logits.flatten())
    
    def get_pred_logit_X(self,x):
        logits = torch.matmul(x,self.coefs_X) + self.intercept_X
        return torch.sigmoid(logits.flatten())
        
        
    def get_pred_X(self,X,Y,S): 
        X_not = torch.cat((X[:,:self.X_target],X[:,(self.X_target+1):]),1)
        feats = torch.cat((X_not , (-1*Y+1)*S, Y*S), 1).float()
        if(self.model_type == "mlp"):
            preds = self.get_pred_mlp_X(feats)
        else:
            preds = self.get_pred_logit_X(feats)
        return preds
    
    def get_pred_Y(self,X,Y,S):
        if(self.start_X_primary > 1):
            S_X = torch.ones(self.start_X_primary)
            S_X[self.X_target] = 0
            X_gene = X[:,:self.start_X_primary]
            X_primary = X[:,self.start_X_primary:]
            feats = torch.cat(((-1*X_gene+1)*S_X, X_gene*S_X  , X_primary,(-1*Y+1)*S, Y*S), 1).float()
        else:
            S_X = torch.ones(self.X_dim)
            S_X[self.X_target] = 0
            feats = torch.cat(((-1*X+1)*S_X, X*S_X  , (-1*Y+1)*S, Y*S), 1).float()
        if(self.model_type == "mlp"):
            preds = self.get_pred_mlp_Y(feats)
        else:
            preds = self.get_pred_logit_Y(feats)
        return preds
    
    def get_gcm(self,X,Y,S):
        preds_X = self.get_pred_X(X,Y,S)
        preds_Y = self.get_pred_Y(X,Y,S)
        X_j = X[:,X_target]
        Y_k = Y[:,Y_target]
        T = gcm(X_j,Y_k, preds_X, preds_Y)
        p = 2*(1 - norm.cdf(T.detach().numpy()))
        return T, p 
    
    def linear_probs(self,S_out,S_iter):
        self.weight[:S_iter,S_iter]
        self.bias[S_iter]
        if(S_iter > 0):
            S = S_out[:,0]
            logprob = self.bias[S_iter] + torch.dot(S,self.weight[:S_iter,S_iter])
        else:
            logprob = self.bias[S_iter]
        logits = torch.stack((-logprob,logprob))
        S_add = gumbel_softmax(logits,0.1,1,2,hard=True).view(-1,2)
        if S_iter == self.Y_target:
            S_add[0][0] = 1
            S_add[0][1] = 0
        return S_add, logits
    
    def forward_univariate(self,X,Y,temp):
        S_out = gumbel_softmax(self.logs,temp,self.Y_dim,2,hard=True).view(-1,2)
        S_not = S_out[:,0]
        S = S_out[:,1]
        S[Y_target] = 0 
        S_not[Y_target] = 1
        T,p = self.get_gcm(X,Y,S)
        probs = torch.nn.functional.softmax(self.logs,dim=1)[:,1].detach().numpy()
        return T, S, p, probs, self.logs
    
    def forward_linear(self,X,Y,temp):
        S_out = torch.empty(0)
        logit_out = torch.empty(0)
        for S_iter in range(self.Y_dim):
            S_add,logits = self.linear_probs(S_out,S_iter)
            probs = torch.nn.functional.softmax(logits,dim=0).detach().numpy()
            if S_iter == 0:
                S_out = S_add
                probs_out = probs
                logits_out = logits.detach().numpy()
            else:
                S_out = torch.cat((S_out,S_add),dim=0)
                probs_out = np.vstack((probs_out,probs))
                logits_out = np.vstack((logits_out,logits.detach().numpy()))
        S = S_out[:,1]
        T,p = self.get_gcm(X,Y,S)
        return T, S,p, probs_out[:,0], logits_out
    
    def forward(self,X,Y,temp):
        if self.search_type == "univariate":
            T, S,p,probs,logits = self.forward_univariate(X,Y,temp)
        else:
            T, S,p,probs,logits = self.forward_linear(X,Y,temp)
        return T,S,p,probs, logits

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

warnings.filterwarnings("ignore")


resdir = 'results/'
datadir = 'data/'
modeldir = 'models/'

parser = argparse.ArgumentParser(description='parameters for pvalue search.')
parser.add_argument('--X_target',type=int,default =0)
parser.add_argument('--Y_target',type=int,default =0)
parser.add_argument('--NUM_EPOCHS',type=int,default =1000)
parser.add_argument('--r',type=float,default =0.005)
parser.add_argument('--learning_r',type=float,default =0.5)
parser.add_argument('--stop_pval',type=float,default =0.5)
parser.add_argument('--stop_train',type=float,default =100)
parser.add_argument('--model_type',type=str,default ='logit')
parser.add_argument('--label',type=str,default ='synth_med')
parser.add_argument('--search_type',type=str,default ='univariate')
parser.add_argument("--exhaustive", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Exhaustively compute subsets.")
parser.add_argument('--start_X_primary',type=int,default =-1)
parser.add_argument('--subset_primary',type=str,default =None)
args = parser.parse_args()
    
label = args.label
model_type = args.model_type
X_target= args.X_target
Y_target= args.Y_target
NUM_EPOCHS = args.NUM_EPOCHS
r = args.r
learning_r =args.learning_r
stop_pval = args.stop_pval
stop_train = args.stop_train
exhaustive = args.exhaustive
search_type = args.search_type
subset_primary = args.subset_primary
start_X_primary = args.start_X_primary

x_ds = label +'_x.npy'
y_ds = label +'_y.npy'
X = torch.from_numpy(np.load(datadir + x_ds))
Y = torch.from_numpy(np.load(datadir + y_ds))

label_granular =  label
if(subset_primary != None):
    print("Subset: " + subset_primary)
    prim = pd.read_pickle(datadir + label + "_primary.pkl")  
    sub = np.where(prim == subset_primary)[0]
    X = X[sub,:]
    Y = Y[sub,:]
    label_granular = label + "-" + subset_primary

if(model_type == "mlp"):
    model_x = torch.load(modeldir +"Predictive Models/" +label + "_torchmodel" + "_xmodel_" + str(X_target) + ".pkl")
    model_y = torch.load(modeldir +"Predictive Models/" +label + "_torchmodel" + "_ymodel_" + str(Y_target) + ".pkl")
else:
    model_y = pickle.load(open(modeldir +"Predictive Models/" +label + "_logit" + "_ymodel_" + str(Y_target) + "_0.2_20_1.0.pkl", 'rb'))
    model_x = pickle.load(open(modeldir +"Predictive Models/" +label + "_logit" +  "_xmodel_" + str(X_target) + "_0.2_20_1.0.pkl", 'rb'))

subset_pick = subset_optim(0.5,X.shape[1],Y.shape[1],model_x,model_y,X_target,Y_target,model_type = model_type, search_type = search_type, start_X_primary=start_X_primary)

opt = torch.optim.Adam(subset_pick.parameters(), lr=learning_r)
T_list = []
S_list = []
p_list = []
prob_list = []
duration_list = []
early_stop = False

for epoch in range(NUM_EPOCHS):
    start = time.time()
    # set to train mode (so the weights update)
    subset_pick.train()
    opt.zero_grad()
    T, S, p,probs,logits = subset_pick.forward(X,Y,np.exp(-r*epoch))
    T.backward()
    opt.step()
    T_list.append(float(T))
    S_list.append(np.where(S.detach().numpy())[0].tolist())
    prob_list.append(probs)
    p_list.append(p)
    duration_list.append(time.time() - start)
    if p > stop_pval:
        early_stop = True
        break
        
ds_gso = pd.DataFrame(T_list,columns=["T"])
ds_gso['S'] = [tuple(f) for f in S_list]
ds_gso['p'] = p_list
ds_gso['prob'] = prob_list
ds_gso['duration'] = duration_list

if(early_stop == True):
    print("Early Stop Determined")
    ds_comb = ds_gso


if(early_stop == False):   
    print("Step 1 Complete: GS Optimization")
    possible_vars = [x for x in range(0,Y.shape[1]) if x !=Y_target]
    set_list = []
    possible_vars = [x for x in range(0,Y.shape[1]) if x !=Y_target]
    for L in range(0, len(possible_vars)+1):
        for cond_set in itertools.combinations(possible_vars, L):
            set_list.append(cond_set)

    S_track = list(set([tuple(f) for f in S_list[:stop_train]]))
    set_list = [i for i in set_list if i not in S_track]
    weight_list = np.array([get_weights(t,prob_list[stop_train-1]) for t in set_list])
    if 0 in weight_list:
        bottom = 1/len(weight_list)/1e40
        weight_list[weight_list < bottom] = bottom
    weight_list = weight_list/ np.sum(weight_list)

    T_list = []
    S_list = []
    p_list = []
    prob_list = []
    duration_list = []
    for x in np.random.choice(set_list,size=min(len(weight_list),NUM_EPOCHS-stop_train),replace=False,p=weight_list):
        start = time.time()
        S = torch.zeros(Y.shape[1])
        S[[x]] = 1
        T, p = subset_pick.get_gcm(X,Y,S)
        T = T.detach().numpy().item()
        duration = time.time() - start
        T_list.append(T)
        S_list.append(x)
        prob_list.append([])
        p_list.append(p)
        duration_list.append(time.time() - start)
        if p > stop_pval:
            print("Early Stop Determined")
            early_stop = True
            break

    ds_hybrid = pd.DataFrame(T_list,columns=["T"])
    ds_hybrid['S'] = [tuple(f) for f in S_list]
    ds_hybrid['p'] = p_list
    ds_hybrid['prob'] = prob_list
    ds_hybrid['duration'] = duration_list
    ds_hybrid = pd.concat([ds_gso.loc[:stop_train-1],ds_hybrid]).reset_index(drop=True)
    
    ds_exhaustive = ds_hybrid.drop_duplicates(subset=['T','S','p'],ignore_index=True)
    ds_gso['Type'] = "GSO"
    ds_hybrid['Type'] = "Hybrid"
    ds_comb = pd.concat([ds_gso,ds_hybrid]).reset_index()
    ds_comb = ds_comb.rename(columns={"index": "iter"})
    ds_comb['Dataset'] = label
    ds_comb['X_target'] = X_target
    ds_comb['Y_target'] = Y_target
    ds_comb['Model Type'] = model_type
    ds_comb['stop_train'] = 100
    ds_comb['stop_pval'] =  0.8
    ds_comb['NUM_EPOCHS']= 1000
    ds_comb['LR'] = 1e-1
    ds_comb['r'] = 0.005
    ds_comb.to_pickle(resdir + label_granular.replace('_', '') + "_" + model_type + "_"+ search_type + "_" + str(X_target) + "_" + str(Y_target) + "_combined.pkl")
    print("Step 2 Complete: Random Search")
    
    if((early_stop == False) & (exhaustive == True)): 
        print(exhaustive)
        print(early_stop)
        print((early_stop == False) & (exhaustive == True))
        S_track = list(set([tuple(f) for f in ds_exhaustive["S"]]))
        set_list = [i for i in set_list if i not in S_track]
        T_list = []
        S_list = []
        p_list = []
        prob_list = []
        duration_list = []

        for x in set_list:
            start = time.time()
            S = torch.zeros(Y.shape[1])
            S[[x]] = 1
            T, p = subset_pick.get_gcm(X,Y,S)
            T = T.detach().numpy().item()
            duration = time.time() - start
            T_list.append(T)
            S_list.append(x)
            prob_list.append([])
            p_list.append(p)
            duration_list.append(time.time() - start)

        ds_add = pd.DataFrame(T_list,columns=["T"])
        ds_add['S'] = [tuple(f) for f in S_list]
        ds_add['p'] = p_list
        ds_add['prob'] = prob_list
        ds_add['duration'] = duration_list
        ds_exhaustive = pd.concat([ds_exhaustive,ds_add]).sample(frac=1).reset_index(drop=True)
        ds_exhaustive['p_rank'] = ds_exhaustive["T"].rank()
        ds_gso = ds_gso.merge(ds_exhaustive[["S","p_rank"]], how="left", on="S")
        ds_hybrid = ds_hybrid.merge(ds_exhaustive[["S","p_rank"]], how="left", on="S")

        ds_exhaustive['p_rank_cum'] = ds_exhaustive['p_rank'].cummin()
        ds_gso['p_rank_cum'] = ds_gso['p_rank'].cummin()
        ds_hybrid['p_rank_cum'] = ds_hybrid['p_rank'].cummin()

        ds_exhaustive['Type'] = "Exhaustive"
        ds_gso['Type'] = "GSO"
        ds_hybrid['Type'] = "Hybrid"
        ds_comb = pd.concat([ds_exhaustive,ds_gso,ds_hybrid]).reset_index()
        ds_comb = ds_comb.rename(columns={"index": "iter"})
        print("Step 3 Complete: Exhaustive Search")


ds_comb['Dataset'] = label
ds_comb['X_target'] = X_target
ds_comb['Y_target'] = X_target
ds_comb['Model Type'] = model_type
ds_comb['Search Type'] = search_type
ds_comb['stop_train'] = stop_train
ds_comb['stop_pval'] = stop_pval
ds_comb['NUM_EPOCHS']= NUM_EPOCHS
ds_comb['LR'] = learning_r
ds_comb['r'] = r
ds_comb.to_pickle(resdir + label_granular.replace('_', '') + "_" + model_type + "_"+ search_type + "_" + str(X_target) + "_" + str(Y_target) + "_combined.pkl")