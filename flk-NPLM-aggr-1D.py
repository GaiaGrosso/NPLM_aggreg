import numpy as np
import os, time
import torch

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
from scipy.stats import norm, expon
import os, json, glob, random, h5py

from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest

from FLKutils import *

# multiple testing definition
flk_sigmas = [0.1, 0.3, 0.7, 1.4, 3.0]
M          = 1000
lam        = 1e-6
Ntoys      = 100

# initialize 
tstat_dict = {}
seeds_dict = {}
seeds_flk_dict = {}

# problem definition
N_ref      = 200000
N_Bkg      = 2000*4
N_Sig      = 13*4
z_ratio    = N_Bkg*1./N_ref
Sig_loc    = 4
Sig_std    = 0.64
extra_flat_dimensions = 0

folder_out = '/n/home00/ggrosso/1D-EXPO/out-NPLM-store/'
NP = 'Ntoys%i_NR%i_NB%i_NS%i_loc%s_std%s'%(Ntoys,N_ref, N_Bkg, N_Sig, str(Sig_loc), str(Sig_std))
#NP = 'ref_Ntoys%i'%(Ntoys)
if not os.path.exists(folder_out+NP):
    os.makedirs(folder_out+NP)
tstat_dict[NP] = {}
seeds = np.arange(Ntoys)*int(time.time()/1000000)
seeds_flk_dict[NP] = {}
for flk_sigma in flk_sigmas:
    flk_config = get_logflk_config(M,flk_sigma,[lam],weight=z_ratio,iter=[1000000],seed=None,cpu=False)
    t_list = [] 
    seeds_flk = []
    for i in range(Ntoys):
        seed = seeds[i]
        np.random.seed(seed)
        # data
        N_Bkg_Pois  = np.random.poisson(lam=N_Bkg, size=1)[0]
        N_ref_Pois  = np.random.poisson(lam=N_ref, size=1)[0]
        N_observed_ref = N_ref_Pois
        if N_Sig:
            N_Sig_Pois = np.random.poisson(lam=N_Sig, size=1)[0]

        featureData = np.random.exponential(scale=1, size=(N_Bkg_Pois, 1))
        if N_Sig:
            featureSig  = np.random.normal(loc=Sig_loc, scale=Sig_std, size=(N_Sig_Pois,1))
            featureData = np.concatenate((featureData, featureSig), axis=0)
        featureRef  = np.random.exponential(scale=1, size=(N_ref_Pois, 1))
        feature     = np.concatenate((featureData, featureRef), axis=0)

        for i in range(extra_flat_dimensions):
            flat = np.random.uniform(size=(feature.shape[0],1))
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
        #if not i%20: plot_reco=True
        #else: plot_reco=False
        plot_reco=False
        seed_flk = seed*int(time.time())
        seeds_flk.append(seed_flk)
        t_list.append(run_toy(NP, feature, target[:, 0:1],  weight=z_ratio, seed=seed_flk, flk_config=flk_config, output_path='./', plot=plot_reco, df=10)[0])
    tstat_dict[NP][str(flk_sigma)]=np.array(t_list)
    seeds_flk_dict[NP][str(flk_sigma)]=np.array(seeds_flk)

f = h5py.File('%s/%s/%s.h5'%(folder_out, NP, NP), 'w')
for flk_sigma in flk_sigmas:
  f.create_dataset(str(flk_sigma), data=tstat_dict[NP][str(flk_sigma)], compression='gzip')
  f.create_dataset('seed_flk_%s'%(str(flk_sigma)), data=seeds_flk_dict[NP][str(flk_sigma)], compression='gzip')
f.create_dataset('seed_toy', data=seeds, compression='gzip')
f.close()
