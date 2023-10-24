import argparse
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle


# Define directories for data and models
datadir = 'data/'
modeldir = 'models/Predictive Models/'


# Parse command line arguments
# X_target - Index of target feature in the X dataset
# mask_p - Probability of masking (portion of data to be masked)
# niter_mask - Number of iterations for generating masks
# split_size - Amount of data to be used for training (can be 1.0 when using GCM test statistic)
# label - Dataset name
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--X_target',type=int,default =0)
parser.add_argument('--mask_p',type=float,default =0.2)
parser.add_argument('--hidden_layers',type=str,default ='200,200')
parser.add_argument('--niter_mask',type=int,default =20)
parser.add_argument('--split_size',type=float,default =1.0)
parser.add_argument('--label',type=str,default = "regular")
args = parser.parse_args()



X_target = args.X_target
mask_p = args.mask_p
niter_mask = args.niter_mask
hiddens  = list(map(int, args.hidden_layers.split(',')))
split_size = args.split_size
label = args.label
x_ds = label + "_x.npy"
y_ds = label + "_y.npy"

X = np.load(datadir + x_ds)
Y = np.load(datadir + y_ds)


# Prepare data for training
n = X.shape[0]
num_S2 = Y.shape[1]
train_int = int(split_size*X.shape[0])

X_train = X[:train_int]
Y_train = Y[:train_int]


# Generate masks and features for training
masks = np.random.binomial(1,mask_p,size=num_S2*niter_mask*train_int)
masks = np.reshape(masks,[train_int*niter_mask, num_S2])
ints = np.random.choice(train_int, size=niter_mask*train_int, replace=True)

Y_train = 2*Y_train -1
feat = np.hstack((np.delete(X_train[ints,:],X_target,1),Y_train[ints,:]*masks == -1,Y_train[ints,:]*masks == 1))
outputs = X_train[ints,X_target]


# Initialize and train machine learning models
mlp = MLPClassifier(random_state=1, solver='adam',max_iter=1000,hidden_layer_sizes=hiddens)
rf = RandomForestClassifier(n_estimators = 500, random_state = 42,max_depth=10)
lg = sklearn.linear_model.LogisticRegression(random_state=0,solver='lbfgs')

model_mlp = mlp.fit(feat,outputs)
model_rf = rf.fit(feat, outputs)
model_logit = lg.fit(feat, outputs)

filename = modeldir + label + "_" + "mlp_xmodel_" + str(X_target) + "_" + str(mask_p) + "_" +str(niter_mask) + "_" + str(split_size)+".pkl"
pickle.dump(model_mlp, open(filename, 'wb'))

filename = modeldir + label + "_" + "rf_xmodel_"  + str(X_target) + "_" + str(mask_p) + "_" +str(niter_mask) + "_" + str(split_size)+".pkl"
pickle.dump(model_rf, open(filename, 'wb'))

filename = modeldir + label + "_" +  "logit_xmodel_"  + str(X_target) + "_" + str(mask_p) + "_" +str(niter_mask) + "_" + str(split_size)+".pkl"
pickle.dump(model_logit, open(filename, 'wb'))
