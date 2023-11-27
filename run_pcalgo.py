from gcm_funs import *


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


X = np.load(datadir + x_ds)
Y = np.load(datadir + y_ds)


num_S1 = X.shape[1]
num_S2 = Y.shape[1]
duration_mat =  np.zeros((num_S1,num_S2))
ajac_mat = np.ones((num_S1,num_S2))
p_mat = np.zeros((num_S1,num_S2))

if(model_type == "rf"):
    clf_X = RandomForestClassifier(n_estimators = 500, random_state = 42,max_depth=10)
    clf_Y = RandomForestClassifier(n_estimators = 500, random_state = 42,max_depth=10)
elif(model_type == "logit"):
    clf_X = sklearn.linear_model.LogisticRegression(random_state=0,solver='lbfgs')
    clf_Y = sklearn.linear_model.LogisticRegression(random_state=0,solver='lbfgs')
elif(model_type == "mlp"):
    clf_X = MLPClassifier(random_state=1, solver='adam',max_iter=1000,hidden_layer_sizes=(200,200))
    clf_Y = MLPClassifier(random_state=1, solver='adam',max_iter=1000,hidden_layer_sizes=(200,200))


for L in range(num_S2 - 1):
    print(L)
    for X_target in range(num_S1):
        for Y_target in range(num_S2):
            start = time.time()
            possible_vars = np.where(ajac_mat[X_target,:]==1)[0]
            possible_vars = [x for x in possible_vars if x !=Y_target]
            for cond_set in itertools.combinations(possible_vars, L):
                #print(str(X_target)+"_" + str(Y_target)+ "_" + str(p))
                T,p = gcm_bespoke(clf_X,clf_Y,X,Y,X_target, Y_target,cond_set)
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