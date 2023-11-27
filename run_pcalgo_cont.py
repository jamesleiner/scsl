from gcm_funs import *
def gcm_regress(clf_X,clf_Y,X,Y,X_target, Y_target,cond_set):
    X_j = X[:,X_target]
    Y_k = Y[:,Y_target]
    X_notJ = np.delete(X,X_target,1)
    feats = np.concatenate((X_notJ,Y[:,cond_set]),axis=1)
    clf_X = clf_X.fit(feats,X_j)
    clf_Y = clf_Y.fit(feats,Y_k)
    pred_Xj = clf_X.predict_proba(feats)[:,1]
    pred_Yk = clf_Y.predict(feats)
    T,p = gcm(X_j,Y_k, pred_Xj, pred_Yk)
    return T,p


datadir = '/ocean/projects/mth200005p/jleiner/hrt-factor/data/'
modeldir = '/ocean/projects/mth200005p/jleiner/hrt-factor/models/'
resdir = '/ocean/projects/mth200005p/jleiner/hrt-factor/results/'

parser = argparse.ArgumentParser(description='Parameters for GCM.')
parser.add_argument('--label',type=str,default ='synth_med')
parser.add_argument('--model_type',type=str,default ='rf')
parser.add_argument('--alpha',type=float,default =0.2)
args = parser.parse_args()

label = args.label   
alpha = args.alpha  
model_type = args.model_type
x_ds = label + '_x.npy'
y_ds  =label + '_y.npy'

X = np.load(datadir + x_ds)
Y = np.load(datadir + y_ds)


num_S1 = X.shape[1]
num_S2 = Y.shape[1]
duration_mat =  np.zeros((num_S1,num_S2))
ajac_mat = np.ones((num_S1,num_S2))
p_mat = np.zeros((num_S1,num_S2))


#regress_X = sklearn.linear_model.LogisticRegression(random_state=0,solver='lbfgs')
#regress_Y = sklearn.linear_model.LinearRegression()

regress_X = MLPClassifier(random_state=1, solver='adam',max_iter=1000,hidden_layer_sizes=(200,200))
regress_Y = sklearn.neural_network.MLPRegressor(random_state=1, solver='adam',max_iter=1000,hidden_layer_sizes=(200,200))


for L in range(num_S2 - 1):
    print(L)
    for X_target in range(num_S1):
        for Y_target in range(num_S2):
            start = time.time()
            possible_vars = np.where(ajac_mat[X_target,:]==1)[0]
            possible_vars = [x for x in possible_vars if x !=Y_target]
            for cond_set in itertools.combinations(possible_vars, L):
                T,p = gcm_regress(regress_X,regress_Y,X,Y,X_target, Y_target,cond_set)
                print(p)
                if(p > p_mat[X_target,Y_target]):
                    p_mat[X_target,Y_target] = p
                if(p > alpha):
                    ajac_mat[X_target,Y_target] = 0
                    break
            duration_mat[X_target,Y_target] = duration_mat[X_target,Y_target] + time.time() - start
    
results = []
for X_target in range(num_S1):
    for Y_target in range(num_S2):
        results.append([label,X_target, Y_target, alpha,model_type,p_mat[X_target,Y_target],ajac_mat[X_target,Y_target],duration_mat[X_target,Y_target]])
results = pd.DataFrame(results, columns = ['Dataset','X_target','Y_target',"alpha","Model Type","pvalue","in_graph","duration"])
results.to_pickle(resdir + label + "_" + model_type + "_" + str(alpha) + "_pcalgo.pkl")