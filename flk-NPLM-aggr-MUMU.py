import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
font = font_manager.FontProperties(family='serif', size=20)

import os, json, glob, random, h5py, time, torch

from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest

from FLKutils import *
from SampleUtils import *

# multiple testing definition
flk_sigmas = []
sigma_perc_list = [1, 5, 25, 50, 75, 95, 99]
M          = 10000
lam        = 1e-6
Ntoys      = 100

# initialize 
tstat_dict = {}
seeds_dict = {}
seeds_flk_dict = {}

## GLOBAL VARIABLES ##
INPUT_PATH_BKG = '/n/home00/ggrosso/MUMU/DiLepton_SM/'
INPUT_PATH_SIG = '/n/home00/ggrosso/MUMU/DiLepton_Zprime600/'#M180_W2e-02/'

# problem definition
features_read = ['pt1', 'pt2', 'eta1', 'eta2', 'delta_phi', 'mll']
features_tr = ['pt1', 'pt2', 'eta1', 'eta2', 'delta_phi']
NS = int(12)
NB = int(20000)
NR = int(100000)
weight_Ref = NB*1./NR
Pois_ON=True
print('NS=%i, NB=%i, NR=%i'%(NS, NB, NR))

N_ref      = NR
N_Bkg      = NB
N_Sig      = NS
z_ratio    = N_Bkg*1./N_ref

# physics selections                                                                                                                                                                         
M_cut   = 60
PT_cut  = 20.
ETA_cut = 2.4

seeds = np.arange(Ntoys)*int(time.time()/10000000)

# read data all
data_BR_all = BuildSample_DY(N_Events=-1, INPUT_PATH=INPUT_PATH_BKG, features=features_read, seed=1234, nfiles=66, shuffle=False)
data_S_all = BuildSample_DY(N_Events=-1, INPUT_PATH=INPUT_PATH_SIG, features=features_read, seed=1234, nfiles=1, shuffle=False)

# compute std, mean R (use sample after physics selections are applied: data_R)
data_dict = {}
for i in range(len(features_read)): data_dict[features_read[i]] = data_BR_all[:, i]
mask = 1*(data_dict['mll']>=M_cut)*(np.abs(data_dict['eta1'])<ETA_cut)*(np.abs(data_dict['eta2'])<ETA_cut)*(data_dict['pt1']>=PT_cut)*(data_dict['pt2']>=PT_cut)
data_R = data_BR_all[mask>0]
data_R = data_R[:, :len(features_tr)]
mean_R = np.mean(data_R, axis=0)
std_R  = np.std(data_R, axis=0)

# compute flk sigma candidates (use sample after physics selections are applied: data_R)
data_R = data_R[:20000, :]
print(data_R.shape)
data_R = standardize(data_R, mean_R, std_R)
#flk_sigmas = [0.2, 0.4, 0.6, 1, 1.2]#[0.5, 0.7, 1, 4, 6]
flk_sigmas = [candidate_sigma(data_R, perc=i) for i in sigma_perc_list]
print('sigmas: ', flk_sigmas)
for flk_sigma in flk_sigmas:
    tstat_dict[str(flk_sigma)] = np.array([])
    seeds_flk_dict[str(flk_sigma)]= np.array([])

# define output path and make directory if needed
folder_out = '/n/home00/ggrosso/MUMU/out/'
if NS:
    folder_out += INPUT_PATH_SIG.split('/')[-2]
    folder_out += '/'
NP = 'MUMU%iD_NR%i_NB%i_NS%i_M%i_lam%s_sig'%(len(features_tr), NR, NB, NS, M, str(lam))

for flk_sigma in flk_sigmas:
    NP +='-'+str(flk_sigma)

if not os.path.exists(folder_out+NP):
    os.makedirs(folder_out+NP)
print('Save at: ', folder_out+NP)

# run toys
idx_BR = np.arange(data_BR_all.shape[0])
idx_S = np.arange(data_S_all.shape[0])
for i in range(Ntoys):
    seed = seeds[i]
    np.random.seed(seed)
    # data
    NS_p = NS
    NB_p = NB
    if Pois_ON:
        NS_p = np.random.poisson(lam=NS, size=1)[0]
        NB_p = np.random.poisson(lam=NB, size=1)[0]
    np.random.shuffle(idx_BR)
    np.random.shuffle(idx_S)
    data_BR = data_BR_all[idx_BR[:NR+NB_p], :]
    data_S  = data_S_all[idx_S[:NS_p], :]    
    label_R = np.zeros((NR,1))
    label_D = np.ones((NB_p+NS_p, 1))
    target  = np.concatenate((label_D,label_R), axis=0)
    feature = np.concatenate((data_S, data_BR), axis=0)
    mask_dict = {}
    for j in range(len(features_read)):
        mask_dict[features_read[j]] = feature[:, j]
    mask = 1*(mask_dict['mll']>=M_cut)*(np.abs(mask_dict['eta1'])<ETA_cut)*(np.abs(mask_dict['eta2'])<ETA_cut)*(mask_dict['pt1']>=PT_cut)*(mask_dict['pt2']>=PT_cut)
    
    feature = feature[mask>0]
    target  = target[mask>0]
    feature_tr = feature[:, :len(features_tr)]
    feature_tr = standardize(feature_tr, mean_R, std_R)
    # run
    plot_reco=False
    verbose=False
    if not i%int(Ntoys/5):
        verbose=True
        plot_reco=True
        print('plot toy', i)
    for flk_sigma in flk_sigmas:
        flk_config = get_logflk_config(M,flk_sigma,[lam],weight=weight_Ref,iter=[1000000],seed=None,cpu=False)
        seed_flk = seed*int(flk_sigma*10)
        t = run_toy(NP+'_sig'+str(flk_sigma), feature_tr, target, weight=weight_Ref, seed=seed_flk, flk_config=flk_config,
                    savefig=plot_reco, output_path='%s/%s/'%(folder_out, NP),  plot=plot_reco, feature_reco=feature,
                    verbose=verbose)
        tstat_dict[str(flk_sigma)]=np.append(tstat_dict[str(flk_sigma)], t)
        seeds_flk_dict[str(flk_sigma)]=np.append(seeds_flk_dict[str(flk_sigma)], seed_flk)
        
tstat_past_dict     = {}
seeds_flk_past_dict = {}
seeds_past          = np.array([])

if os.path.exists('%s/%s/tvalues.h5'%(folder_out, NP)):
    print('collecting previous tvalues')
    f = h5py.File('%s/%s/tvalues.h5'%(folder_out, NP), 'r')
    seeds_past = np.array(f.get('seed_toy'))
    for flk_sigma in flk_sigmas:
        tstat_past_dict[str(flk_sigma)] = np.array(f.get(str(flk_sigma) ) )
        seeds_flk_past_dict[str(flk_sigma)] = np.array(f.get('seed_flk_%s'%(str(flk_sigma)) ) )
    f.close()
else:
    for flk_sigma in flk_sigmas:
        tstat_past_dict[str(flk_sigma)] = np.array([])
        seeds_flk_past_dict[str(flk_sigma)] = np.array([])

f = h5py.File('%s/%s/tvalues.h5'%(folder_out, NP), 'w')
for flk_sigma in flk_sigmas:
    f.create_dataset(str(flk_sigma), data=np.append(tstat_past_dict[str(flk_sigma)], tstat_dict[str(flk_sigma)]), compression='gzip')
    f.create_dataset('seed_flk_%s'%(str(flk_sigma)), data=np.append(seeds_flk_past_dict[str(flk_sigma)], seeds_flk_dict[str(flk_sigma)]), compression='gzip')
f.create_dataset('seed_toy', data=np.append(seeds_past, seeds), compression='gzip')
f.close()
'''
f = h5py.File('%s/%s/tvalues.h5'%(folder_out, NP), 'w')
for flk_sigma in flk_sigmas:
  f.create_dataset(str(flk_sigma), data=tstat_dict[str(flk_sigma)], compression='gzip')
  f.create_dataset('seed_flk_%s'%(str(flk_sigma)), data=seeds_flk_dict[str(flk_sigma)], compression='gzip')
f.create_dataset('seed_toy', data=seeds, compression='gzip')
f.close()
'''
