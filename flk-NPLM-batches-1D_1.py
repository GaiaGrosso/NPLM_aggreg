import numpy as np
import os, time
import torch

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

from FLKutils import *


# multiple testing definition
n_aggreg = 8
flk_sigmas_perc = [5, 25, 50, 75, 95]#[0.1, 0.3, 0.7, 1.4, 3.0]
M          = 1000
lam        = 1e-6
Ntoys      = 100

# initialize 
tsum_dict   = {}
taggrD_dict = {}
taggrR_dict = {}
seeds_dict  = {}
seeds_flk_dict = {}

# problem definition
NR      = 200000
NB      = 2000
NS      = 13
z_ratio    = NB*1./NR
weight_Ref = z_ratio
Sig_loc    = 4
Sig_std    = 0.64
Pois_ON =True
folder_out = '/n/home00/ggrosso/1D-EXPO/out-NPLM-store/saveD/'
if not os.path.exists(folder_out):
    os.makedirs(folder_out)
if NS:    
    NP = '1D-EXPO_NR%i_NB%i_NS%i_Sloc%s_Sstd%s_M%i_lam%s_naggr%i'%(NR, NB, NS, str(Sig_loc), str(Sig_std), M, str(lam), n_aggreg)
else:
    NP = '1D-EXPO_NR%i_NB%i_NS%i_M%i_lam%s_naggr%i'%(NR, NB, NS, M, str(lam), n_aggreg)
if not os.path.exists(folder_out+NP+'/tmp/'):
    os.makedirs(folder_out+NP+'/tmp/')
seeds = np.arange(Ntoys)*int(time.time()/1000000)
df = 10

np.random.seed(1234)
data_R = np.random.exponential(scale=1, size=(NR, 1))
mean_R = np.mean(data_R, axis=0)
std_R  = np.std(data_R, axis=0)

feat_R = data_R[:10000, :]
feat_R = standardize(feat_R, mean_R, std_R)
flk_sigmas = [candidate_sigma(feat_R, perc=i) for i in flk_sigmas_perc] 
print('sigmas: ', flk_sigmas)
for flk_sigma in flk_sigmas:
    tsum_dict[str(flk_sigma)] = np.array([])
    taggrD_dict[str(flk_sigma)] = np.array([])
    taggrR_dict[str(flk_sigma)] = np.array([])
    seeds_flk_dict[str(flk_sigma)]= np.array([])
            

for i in range(Ntoys):
    seed = seeds[i]
    np.random.seed(seed)
    feature_all = []
    target_all = []
    for n in range(n_aggreg):
        # data
        NS_p = NS
        NB_p = NB
        if Pois_ON:
            NS_p = np.random.poisson(lam=NS, size=1)[0]
            NB_p = np.random.poisson(lam=NB, size=1)[0]
        data_B = np.random.exponential(scale=1, size=(NB_p, 1))
        data_S = np.random.normal(loc=Sig_loc, scale=Sig_std, size=(NS_p,1))
        data_D = np.concatenate((data_B, data_S), axis=0)

        label_R = np.zeros((NR,1))
        label_D = np.ones((NB_p+NS_p, 1))
        target  = np.concatenate((label_D,label_R), axis=0)

        feature = np.concatenate((data_D, data_R), axis=0)
        feature = standardize(feature, mean_R, std_R)
        feature_all.append(feature)
        target_all.append(target)
    # run

    plot_reco=False
    verbose=False
    if not i%20:
        plot_reco=True
        verbose=True
        print(i)
    preds_dict = {}
    for flk_sigma in flk_sigmas:
        if len(tsum_dict[str(flk_sigma)]) and not df: df = np.mean(tsum_dict[str(flk_sigma)])
        flk_config = get_logflk_config(M,flk_sigma,[lam],weight=weight_Ref,iter=[1000000],seed=None,cpu=False)
        seed_flk = seed*int(time.time())
        t_sum, t_aggrD, t_aggrR = run_aggr(n_aggreg, NP+'_sigma%s'%(str(flk_sigma)), feature_all, target_all,  weight=weight_Ref, seed=seed_flk, flk_config=flk_config,
                                           output_path='%s/%s/'%(folder_out, NP), plot=plot_reco, verbose=verbose, savefig=plot_reco, xlabels_list=['x'], yrange_dict={'x': [0, 5]})
        print(t_sum, t_aggrD, t_aggrR )
        tsum_dict[str(flk_sigma)]=np.append(tsum_dict[str(flk_sigma)], t_sum)
        taggrD_dict[str(flk_sigma)]=np.append(taggrD_dict[str(flk_sigma)], t_aggrD)
        taggrR_dict[str(flk_sigma)]=np.append(taggrR_dict[str(flk_sigma)], t_aggrR)
        #preds_dict[str(flk_sigma)]=pred
        seeds_flk_dict[str(flk_sigma)]=np.append(seeds_flk_dict[str(flk_sigma)], seed_flk)
    #if not os.path.exists('%s/%s'%(folder_out, NP)):
    #    os.makedirs('%s/%s'%(folder_out, NP))
    #f = h5py.File('%s/%s/tmp/toy%i_flk%i.h5'%(folder_out, NP, seed, seed_flk), 'w')
    #for flk_sigma in flk_sigmas:
    #    f.create_dataset(str(flk_sigma), data=preds_dict[str(flk_sigma)], compression='gzip')
    #f.close()

tsum_past_dict      = {}
taggregD_past_dict  = {}
taggregR_past_dict  = {}
seeds_flk_past_dict = {}
seeds_past          = np.array([])

if os.path.exists('%s/%s/tvalues.h5'%(folder_out, NP)):
    print('collecting previous tvalues')
    f = h5py.File('%s/%s/tvalues.h5'%(folder_out, NP), 'r')
    seeds_past = np.array(f.get('seed_toy'))
    for flk_sigma in flk_sigmas:
        tsum_past_dict[str(flk_sigma)] = np.array(f.get('tsum_%s'%(str(flk_sigma)) ) )
        taggregR_past_dict[str(flk_sigma)] = np.array(f.get('taggrR_%s'%(str(flk_sigma)) ) )
        taggregD_past_dict[str(flk_sigma)] = np.array(f.get('taggrD_%s'%(str(flk_sigma)) ) )
        seeds_flk_past_dict[str(flk_sigma)] = np.array(f.get('seed_flk_%s'%(str(flk_sigma)) ) )
    f.close()
else:
    for flk_sigma in flk_sigmas:
        tsum_past_dict[str(flk_sigma)] = np.array([])
        taggregR_past_dict[str(flk_sigma)] = np.array([])
        taggregD_past_dict[str(flk_sigma)] = np.array([])
        seeds_flk_past_dict[str(flk_sigma)] = np.array([])

f = h5py.File('%s/%s/tvalues.h5'%(folder_out, NP), 'w')
for flk_sigma in flk_sigmas:
    f.create_dataset('tsum_%s'%(str(flk_sigma)), data=np.append(tsum_past_dict[str(flk_sigma)], tsum_dict[str(flk_sigma)]), compression='gzip')
    f.create_dataset('taggrR_%s'%(str(flk_sigma)), data=np.append(taggregR_past_dict[str(flk_sigma)], taggrR_dict[str(flk_sigma)]), compression='gzip')
    f.create_dataset('taggrD_%s'%(str(flk_sigma)), data=np.append(taggregD_past_dict[str(flk_sigma)], taggrD_dict[str(flk_sigma)]), compression='gzip')
    f.create_dataset('seed_flk_%s'%(str(flk_sigma)), data=np.append(seeds_flk_past_dict[str(flk_sigma)], seeds_flk_dict[str(flk_sigma)]), compression='gzip')
f.create_dataset('seed_toy', data=np.append(seeds_past, seeds), compression='gzip')
f.close()
