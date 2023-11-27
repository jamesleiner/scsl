import os
os.chdir('/jet/home/jleiner/work/py-tetrad/pytetrad')

import argparse
import numpy as np
import pandas as pd
import jpype.imports
import tools.TetradSearch as search
import time 

try:
    jpype.startJVM(classpath=[f"resources/tetrad-current.jar"])
except OSError:
    print("JVM already started")

def compile_results(search,duration,label):
    h=search.get_pcalg()[X.shape[1]:].copy(deep=True)
    h.index =  ["Y" + str(j) for j in range(0,Y.shape[1])]
    h=pd.melt(h[h.columns[:X.shape[1]]],ignore_index=False).reset_index()
    h.columns = ["Y_target","X_target","edge"]
    h["Y_target"] = pd.to_numeric(h["Y_target"].str.slice(1))
    h["X_target"] =  pd.to_numeric(h["X_target"].str.slice(1))
    h['is_edge'] = h.apply(lambda row : row["X_target"] in truth[row["Y_target"]],axis=1)
    h['method'] = label
    h['dataset'] = ds
    h['duration'] = duration
    return(h)

datadir = '/ocean/projects/mth200005p/jleiner/hrt-factor/data/'
resdir = '/ocean/projects/mth200005p/jleiner/hrt-factor/results/'

parser = argparse.ArgumentParser(description='parameters for pvalue search.')
parser.add_argument('--dataset',type=str,default ='synth_med')
parser.add_argument('--method',type=str,default ='fges')
args = parser.parse_args()
ds = args.dataset
method = args.method

X= np.load(datadir+ds +"_x.npy")
Y= np.load(datadir+ds  +"_y.npy")
if "collide" in ds:
    truth = np.load(datadir+ds  + "_xtrue.npy", allow_pickle=True)
else:
    truth = np.load(datadir+ds  + "_ytrue.npy", allow_pickle=True)
    
colnames = ["X" + str(j) for j in range(0,X.shape[1])] + ["Y" + str(j) for j in range(0,Y.shape[1])]
df = pd.DataFrame(np.concatenate((X,Y),axis=1),columns=colnames)
search = search.TetradSearch(df)
search.set_verbose(False)

## Pick the score to use, in this case a continuous linear, Gaussian score.
search.use_conditional_gaussian_score()
search.use_conditional_gaussian_test()
#
# search.use_degenerate_gaussian_score()
# search.use_degenerate_gaussian_test()

## Run various algorithms and print their results. For now (for compability with R)
## Commenting out the ones that won't work with mixed data.

print(method.casefold())
print(ds)

start =time.time()
if method.casefold() == 'fges'.casefold():
    print("FGES")
    search.run_fges()

if method.casefold() == 'boss'.casefold():
    search.run_boss()

if method.casefold() == 'sp'.casefold():
    search.run_sp()
    
if method.casefold() == 'grasp'.casefold():
    search.run_grasp()

if method.casefold() == 'pc'.casefold():
    search.run_pc()

if method.casefold() == 'fci'.casefold():
    search.run_fci()

if method.casefold() == 'gfci'.casefold():
    search.run_gfci()

if method.casefold() == 'bfci'.casefold():
    search.run_bfci()
    
if method.casefold() == 'grasp-fci'.casefold():
    search.run_grasp_fci()

if method.casefold() == 'ccd'.casefold():
    search.run_ccd()

duration = time.time() - start
h=search.get_pcalg()[X.shape[1]:].copy(deep=True)
h.index =  ["Y" + str(j) for j in range(0,Y.shape[1])]
h=pd.melt(h[h.columns[:X.shape[1]]],ignore_index=False).reset_index()
h.columns = ["Y_target","X_target","edge"]
h["Y_target"] = pd.to_numeric(h["Y_target"].str.slice(1))
h["X_target"] =  pd.to_numeric(h["X_target"].str.slice(1))
h['is_edge'] = h.apply(lambda row : row["X_target"] in truth[row["Y_target"]],axis=1)
h['method'] = method
h['dataset'] = ds
h['duration'] = duration
h.to_pickle(resdir + ds + '_' + method + '_benchmarks.pkl')