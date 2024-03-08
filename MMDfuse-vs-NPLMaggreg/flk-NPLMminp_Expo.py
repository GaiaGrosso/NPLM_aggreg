import numpy as np
import os, time, argparse, datetime, torch, glob, random, h5py, json

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
from scipy.stats import norm, expon

from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest

from utils.FLKutils import *
from utils.GENutils import *
parser = argparse.ArgumentParser()
parser.add_argument('-j', '--jsonfile', type=str, help="json file", required=True)
parser.add_argument('-t', '--toys', type=int, help="number of toys", required=True)

args = parser.parse_args()

#### set up parameters ###############################                                                                                                                                    
with open(args.jsonfile, 'r') as jsonfile:
    config_json = json.load(jsonfile)

# multiple testing definition                                                                                                                                                             
flk_sigmas = [0.1, 0.3, 0.7, 1.4, 3.0]
M          = 1000
lam        = 1e-6
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
folder_out = config_json["output_directory"]+'/'
print(folder_out)
NP= "M%i_lam%s/"%(M, str(lam))
if not os.path.isdir(folder_out+NP):
    print('mkdir ', folder_out+NP)
    os.makedirs(folder_out+NP)

# initialize 
tstat_dict = {}
seeds_dict = {}
seeds_flk_dict = {}
seed_toys = np.arange(Ntoys)*int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)

for flk_sigma in flk_sigmas:
    flk_config = get_logflk_config(M,flk_sigma,[lam],weight=z_ratio,iter=[1000000],seed=None,cpu=False)
    t_list = [] 
    seeds_flk = []
    for i in range(Ntoys):
        seed = seed_toys[i]
        rng = np.random.default_rng(seed=seed)
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
        seed_flk = int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)
        seeds_flk.append(seed_flk)
        t_list.append(run_toy(NP, feature, target[:, 0:1],  weight=z_ratio, seed=seed_flk, flk_config=flk_config, output_path='./', plot=False, df=10)[0])
    tstat_dict[str(flk_sigma)]=np.array(t_list)
    seeds_flk_dict[str(flk_sigma)]=np.array(seeds_flk)
'''
# collect previous runs
tstat_dict_past = {}
tstat_dict_past['seed_toy'] = np.array([])
for flk_sigma in flk_sigmas:
    tstat_dict_past[str(flk_sigma)] = np.array([])
    tstat_dict_past['seed_flk_%s'%(str(flk_sigma))] = np.array([])
if os.path.exists('%s/%s/tests.h5'%(folder_out, NP)):
    print('file exists. Loading previous experiments.')
    f = h5py.File('%s/%s/tests.h5'%(folder_out, NP), 'r')
    for k in list(f.keys()):
        tstat_dict_past[k] = np.array(f[k])
    f.close()
'''
# save new
tmp_id = int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)
f = h5py.File('%s/%s/%i_tests.h5'%(folder_out, NP, tmp_id), 'w')
for flk_sigma in flk_sigmas:
  f.create_dataset(str(flk_sigma), data=tstat_dict[str(flk_sigma)], compression='gzip')
  f.create_dataset('seed_flk_%s'%(str(flk_sigma)), data=seeds_flk_dict[str(flk_sigma)], compression='gzip')
f.create_dataset('seed_toy', data=seed_toys, compression='gzip')

f.close()
