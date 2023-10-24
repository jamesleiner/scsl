import argparse
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle


datadir = '/ocean/projects/mth200005p/jleiner/hrt-factor/data/'
modeldir = '/ocean/projects/mth200005p/jleiner/hrt-factor/models/Predictive Models/'

# Parse command line arguments
# Y_target - Index of target feature in the Y dataset
# mask_p - Probability of masking (portion of data to be masked)
# niter_mask - Number of iterations for generating masks
# hidden_layers - Hidden layers for MLP models
# split_size - Amount of data to be used for training (can be 1.0 when using GCM test statistic)
# num_X_mask - First X columns will be excluded from coniditoning if used. -1 indicates all X's will be used as features
# label - Dataset name
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--Y_target',type=int,default =0)
parser.add_argument('--mask_p',type=float,default =0.2)
parser.add_argument('--hidden_layers',type=str,default ='200,200')
parser.add_argument('--niter_mask',type=int,default =20)
parser.add_argument('--split_size',type=float,default =1.0)
parser.add_argument('--label',type=str,default = "regular")
parser.add_argument('--num_X_mask',type=int,default = -1)
args = parser.parse_args()

Y_target = args.Y_target
mask_p = args.mask_p
niter_mask = args.niter_mask
hiddens  = list(map(int, args.hidden_layers.split(',')))
split_size = args.split_size
label = args.label
x_ds = label + "_x.npy"
y_ds = label + "_y.npy"
num_X_mask = args.num_X_mask

X = np.load(datadir + x_ds)
Y = np.load(datadir + y_ds)

n = X.shape[0]
num_S2 = Y.shape[1]
train_int = int(split_size*X.shape[0])

X_train = X[:train_int]
Y_train = Y[:train_int]
X_test = X[train_int:]
Y_test = Y[train_int:]

masks = np.random.binomial(1,mask_p,size=num_S2*niter_mask*train_int)
masks = np.reshape(masks,[train_int*niter_mask, num_S2])

#mask target variable in input dataset 
masks[:,Y_target]= 0 
ints = np.random.choice(train_int, size=niter_mask*train_int, replace=True)



#base case
if(num_X_mask < 1):
    Y_train = 2*Y_train -1
    X_train = 2*X_train -1

    choices_X = np.random.choice(X.shape[1],size=niter_mask*train_int,replace=True)
    X_train = X_train[ints,:]
    for i in range(len(choices_X)):
        X_train[i,choices_X[i]] = 0 

    feat = np.hstack((X_train==-1,X_train == 1,Y_train[ints,:]*masks == -1,Y_train[ints,:]*masks == 1))
    outputs = Y_train[ints,Y_target]


#only needed if excluding some X features from consideration
else:
    Y_train = 2*Y_train -1
    X_genes = 2*X_train[:,:num_X_mask] -1
    X_primary = X_train[:,num_X_mask:]
    X_genes = X_genes[ints,:]
    X_primary = X_primary[ints,:]

    choices_X = np.random.choice(X_genes.shape[1],size=niter_mask*train_int,replace=True)
    for i in range(len(choices_X)):
        X_genes[i,choices_X[i]] = 0 

    feat = np.hstack((X_genes==-1,X_genes == 1,X_primary,Y_train[ints,:]*masks == -1,Y_train[ints,:]*masks == 1))
    outputs = Y_train[ints,Y_target]
    

mlp = MLPClassifier(random_state=1, solver='adam',max_iter=1000,hidden_layer_sizes=hiddens)
rf = RandomForestClassifier(n_estimators = 500, random_state = 42,max_depth=10)
lg = sklearn.linear_model.LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000)

model_mlp = mlp.fit(feat,outputs)
model_rf = rf.fit(feat, outputs)
model_logit = lg.fit(feat, outputs)

filename = modeldir + label + "_" + "mlp_ymodel_" + str(Y_target) + "_" + str(mask_p) + "_" +str(niter_mask) + "_" + str(split_size)+".pkl"
pickle.dump(model_mlp, open(filename, 'wb'))

filename = modeldir + label + "_" + "rf_ymodel_"  + str(Y_target) + "_" + str(mask_p) + "_" +str(niter_mask) + "_" + str(split_size)+".pkl"
pickle.dump(model_rf, open(filename, 'wb'))

filename = modeldir + label + "_" +  "logit_ymodel_"  + str(Y_target) + "_" + str(mask_p) + "_" +str(niter_mask) + "_" + str(split_size)+".pkl"
pickle.dump(model_logit, open(filename, 'wb'))
