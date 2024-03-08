import numpy as np
import os, time, argparse, datetime, glob, h5py, random, json

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')

from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest, expon

import jax.numpy as jnp
from jax import random

from utils.MMDFUSEutils import *
from utils.GENutils import *

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--jsonfile', type=str, help="json file", required=True)
parser.add_argument('-t', '--toys', type=int, help="number of toys", required=True)

args = parser.parse_args()

#### set up parameters ###############################                                                                                                  
with open(args.jsonfile, 'r') as jsonfile:
    config_json = json.load(jsonfile)

# multiple testing definition
Ntoys      = args.toys

# problem definition
N_ref      = config_json["N_Ref"]
N_Bkg      = config_json["N_Bkg"]
N_Sig      = config_json["N_Sig"]
z_ratio    = (N_Bkg+N_Sig)*1./N_ref
Sig_loc    = config_json["Sig_loc"]
Sig_std    = config_json["Sig_std"]
is_NP2 = config_json["is_NP2"]
is_Pois = config_json["is_Pois"]
extra_flat_dimensions = config_json["N_flat_extra"]
folder_out = config_json["output_directory"]
#NP = 'MMDfuse_NR%i_NB%i_NS%i_loc%s_std%s'%(N_ref, N_Bkg, N_Sig, str(Sig_loc), str(Sig_std))

t_list = np.array([])
p_list = np.array([])
seed_list = np.array([])
for i in range(Ntoys):
    seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
    rng = np.random.default_rng(seed=seed)
    key = jax.random.PRNGKey(seed)
    # data
    N_Bkg_Pois, N_Sig_Pois = N_Bkg, N_Sig
    if is_Pois:
        N_Bkg_Pois  = rng.poisson(lam=N_Bkg, size=1)[0]
        if N_Sig:
            N_Sig_Pois = rng.poisson(lam=N_Sig, size=1)[0]
    featureData = rng.exponential(scale=1, size=(N_Bkg_Pois, 1))
    if N_Sig:
        if is_NP2:
            featureSig = np.expand_dims(NP2_gen(size=N_Sig_Pois,random_gen=rng), axis=1)*8
        else:
            featureSig  = rng.normal(loc=Sig_loc, scale=Sig_std, size=(N_Sig_Pois, 1))
        featureData = np.concatenate((featureData, featureSig), axis=0)
    featureRef  = rng.exponential(scale=1, size=(N_ref, 1))
    feature     = np.concatenate((featureData, featureRef), axis=0)
    
    for i in range(extra_flat_dimensions):
        flat = rng.uniform(size=(feature.shape[0],1))
        feature = np.concatenate((feature, flat), axis=1)

    # target                                                                                                                         \
    targetData  = np.ones_like(featureData)
    targetRef   = np.zeros_like(featureRef)
    weightsData = np.ones_like(featureData)
    weightsRef  = np.ones_like(featureRef)*z_ratio
    target      = np.concatenate((targetData, targetRef), axis=0)
    weights     = np.concatenate((weightsData, weightsRef), axis=0)
    target      = np.concatenate((target, weights), axis=1)
    
    # run
    X, Y = jnp.array(featureData), jnp.array(featureRef)
    X.device()
    Y.device()
    t1 = time.time()
    output, p_value = mmdfuse(X, Y, key, return_p_val=True)
    t2 = time.time()
    print('Toy %i, p-value: %f, Z-score: %f'%(i, p_value, norm.ppf(1-p_value)))
    print('time: ', round(t2-t1,2))
    t_list = np.append(t_list,output)
    p_list = np.append(p_list,p_value)
    seed_list = np.append(seed_list,seed)
'''
# read previous experiments
if os.path.exists('%s/pvalue.h5'%(folder_out)):
    f = h5py.File('%s/pvalue.h5'%(folder_out), 'r')
    t_list = np.append(np.array(f['p-value']), t_list)
    seed_list= np.append(np.array(f['seed']), seed_list)
    f.close()
'''

# save new file
tmp_id = int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)
f = h5py.File('%s/%i_pvalue.h5'%(folder_out, tmp_id), 'w')
f.create_dataset('p-value', data=p_list, compression='gzip')
f.create_dataset('test', data=t_list, compression='gzip')
f.create_dataset('seed', data=seed_list, compression='gzip')
f.close()
