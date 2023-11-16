import numpy as np
import os, time
import torch
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
font = font_manager.FontProperties(family='serif', size=20)

from scipy.stats import norm, expon
import os, json, glob, random, h5py

from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest

from FLKutils2 import *
from SampleUtils import *
parser   = argparse.ArgumentParser()
parser.add_argument('-s','--NS', type=int, help="number of signal events", required=True)
parser.add_argument('-b','--NB', type=int, help='number of background events',  required=True)
parser.add_argument('-a','--naggr', type=int, help='number of aggregation',  required=True)
parser.add_argument('-t','--toys', type=int, help="number of toys to be processed",   required=False, default = 100)
parser.add_argument('-f','--foldernamesig', type=str, help='name of the signal folder', required=True) 
args     = parser.parse_args()

# multiple testing definition
n_aggreg = args.naggr
flk_sigmas_perc = [5, 25, 50, 75, 95]#[0.1, 0.3, 0.7, 1.4, 3.0]
flk_sigmas_df = [1080, 274, 136, 81, 49]
M          = 10000
lam        = 1e-6
Ntoys      = args.toys#100

## GLOBAL VARIABLES ##
INPUT_PATH_BKG = '/n/home00/ggrosso/MUMU/DiLepton_SM/'
INPUT_PATH_SIG = '/n/home00/ggrosso/MUMU/%s/'%(args.foldernamesig)#DiLepton_Zprime300/'#M180_W2e-02/'

# problem definition
features_read = ['pt1', 'pt2', 'eta1', 'eta2', 'delta_phi', 'mll']
features_tr = ['pt1', 'pt2', 'eta1', 'eta2', 'delta_phi']
yrange_dict = {}
for k in features_read: yrange_dict[k]=[0, 5]
NS = args.NS#int(40*0.25)
NB = args.NB#int(20000*0.25)
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

# output folder
folder_out = '/n/home00/ggrosso/MUMU/out/'
if not os.path.exists(folder_out):
    os.makedirs(folder_out)
if NS:
    folder_out += INPUT_PATH_SIG.split('/')[-2]
    folder_out += '/'
NP = 'v3-MUMU%iD_reffixed_NR%i_NB%i_NS%i_M%i_lam%s_naggr%i'%(len(features_tr), NR, NB, NS, M, str(lam), n_aggreg)

seeds = np.arange(Ntoys)*int(time.time()/1000000)

# read data all
data_B_all = BuildSample_DY(N_Events=-1, INPUT_PATH=INPUT_PATH_BKG, features=features_read, seed=1234, nfiles=66, shuffle=False)
data_S_all = BuildSample_DY(N_Events=-1, INPUT_PATH=INPUT_PATH_SIG, features=features_read, seed=1234, nfiles=1, shuffle=False)
data_R = data_B_all[:NR, :]
data_B_all = data_B_all[NR:, :] 
# compute std, mean R (use sample after physics selections are applied: data_R)
data_dict = {}
for i in range(len(features_read)): data_dict[features_read[i]] = data_R[:, i]
mask = 1*(data_dict['mll']>=M_cut)*(np.abs(data_dict['eta1'])<ETA_cut)*(np.abs(data_dict['eta2'])<ETA_cut)*(data_dict['pt1']>=PT_cut)*(data_dict['pt2']>=PT_cut)
data_R = data_R[mask>0]
#data_R = data_R[:, :len(features_tr)]
mean_R = np.mean(data_R[:len(features_tr)], axis=0)
std_R  = np.std(data_R[:len(features_tr)], axis=0)

# compute flk sigma candidates (use sample after physics selections are applied: data_R)
data_R_small = standardize(data_R[:20000, :], mean_R, std_R)
flk_sigmas = [candidate_sigma(data_R_small, perc=i) for i in flk_sigmas_perc]
print('sigmas: ', flk_sigmas)
del data_R_small

# initialize
tsum_dict  = np.array([])
taggrD_dict = np.array([])
taggrR_dict = np.array([])

# run toys
idx_B = np.arange(data_B_all.shape[0])
idx_S = np.arange(data_S_all.shape[0])
for i in range(Ntoys):
    seed = seeds[i]
    np.random.seed(seed)
    feature_all = []
    feature_all_tr = []
    target_all = []
    for n in range(n_aggreg):
        NS_p = NS
        NB_p = NB
        if Pois_ON:
            NS_p = np.random.poisson(lam=NS, size=1)[0]
            NB_p = np.random.poisson(lam=NB, size=1)[0]
        np.random.shuffle(idx_B)
        np.random.shuffle(idx_S)
        data_B  = data_B_all[idx_B[:NB_p], :]
        data_S  = data_S_all[idx_S[:NS_p], :]
        data_D  = np.concatenate((data_S, data_B), axis=0)

        mask_dict = {}
        for j in range(len(features_read)):
            mask_dict[features_read[j]] = data_D[:, j]
        mask = 1*(mask_dict['mll']>=M_cut)*(np.abs(mask_dict['eta1'])<ETA_cut)*(np.abs(mask_dict['eta2'])<ETA_cut)*(mask_dict['pt1']>=PT_cut)*(mask_dict['pt2']>=PT_cut)
        data_D = data_D[mask>0]
        label_R = np.zeros((data_R.shape[0],1))
        label_D = np.ones((data_D.shape[0], 1))
        target  = np.concatenate((label_D,label_R), axis=0)
        feature = np.concatenate((data_D, data_R), axis=0)
        feature_tr = feature[:, :len(features_tr)]
        feature_tr = standardize(feature_tr, mean_R, std_R)
        feature_all_tr.append(feature_tr)
        feature_all.append(feature)
        target_all.append(target)
    # run
    plot_reco=False
    verbose=False
    if not i%20:
        plot_reco=True
        verbose=True
        print('Toy: ',i)
    preds_dict = {}
    #flk_config = get_logflk_config(M,flk_sigma,[lam],weight=weight_Ref,iter=[1000000],seed=None,cpu=False)
    flk_seeds = [seed*i for i in range(len(flk_sigmas))]
    t_sum, t_aggrD, t_aggrR = run_aggr3(i, n_aggreg, NP, feature_all, target_all,  weight=weight_Ref,
                                        flk_seeds=flk_seeds, flk_sigmas=flk_sigmas, flk_M=M, flk_lam=lam, flk_dfs=flk_sigmas_df,
                                        output_path='%s/%s/'%(folder_out, NP), plot=plot_reco, verbose=verbose, savefig=plot_reco,
                                        xlabels_list=features_read, yrange_dict=yrange_dict
    )
    print(t_sum, t_aggrD, t_aggrR )
    tsum_dict=np.append(tsum_dict, t_sum)
    taggrD_dict=np.append(taggrD_dict, t_aggrD)
    taggrR_dict=np.append(taggrR_dict, t_aggrR)

tsum_past_dict      = np.array([])
taggregD_past_dict  = np.array([])
taggregR_past_dict  = np.array([])
seeds_past          = np.array([])

if os.path.exists('%s/%s/tvalues.h5'%(folder_out, NP)):
    print('collecting previous tvalues')
    f = h5py.File('%s/%s/tvalues.h5'%(folder_out, NP), 'r')
    seeds_past = np.array(f.get('seed_toy'))
    tsum_past_dict = np.array(f.get('tsum') )
    taggregR_past_dict = np.array(f.get('taggrR') )
    taggregD_past_dict = np.array(f.get('taggrD') )
    #seeds_flk_past_dict[str(flk_sigma)] = np.array(f.get('seed_flk_%s'%(str(flk_sigma)) ) )
    f.close()

f = h5py.File('%s/%s/tvalues.h5'%(folder_out, NP), 'w')
f.create_dataset('tsum', data=np.append(tsum_past_dict, tsum_dict), compression='gzip')
f.create_dataset('taggrR', data=np.append(taggregR_past_dict, taggrR_dict), compression='gzip')
f.create_dataset('taggrD', data=np.append(taggregD_past_dict, taggrD_dict), compression='gzip')
#f.create_dataset('seed_flk', data=np.append(seeds_flk_past_dict[str(flk_sigma)], seeds_flk_dict[str(flk_sigma)]), compression='gzip')
f.create_dataset('seed_toy', data=np.append(seeds_past, seeds), compression='gzip')
f.close()
