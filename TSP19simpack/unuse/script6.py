# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:54:50 2019
Analyzed effect of FFT, NOMP (NOMP better, see May19 slides or workflowy)
@author: gupta
"""
import GAutils.master_simulator3 as ms3
import GAutils.config as cfg
import numpy as np
from datetime import date, datetime

def set_it(itrx, xval, idxa, val):
    it_name=['roba','snra','Nsensa','Noba','swidtha']
    it_xlbl = ['Robust level','SNR (dB)','Num. Sensors','Num. Targets','Array width (m)']
    exec("ms3.set_params('"+it_name[itrx]+"',xval)")
    ms3.set_params('Ninst',len(xval))
    ms3.set_params('rng_used',xval)
    ms3.set_params('xlbl',it_xlbl[itrx])
    for i, idx in enumerate(idxa):
        exec("ms3.set_params('"+it_name[idx]+"',np.ones(len(xval), dtype='int')*val[i])")

def run_it(datef, rng, itrx, itry):
    for num, i in enumerate(rng):
        exec("cfg."+itry+"a=i*np.ones(cfg.Ninst, dtype='int')")
        cfg.folder= datef +itrx+'-'+itry+str(i)
        print('Running '+str(num)+'/'+str(len(rng)))
        ms3.main()
   
        
def main():
    datef =('results/'+str(date.today().month)+'_'+str(date.today().day)
        +'_'+str(datetime.now().hour)+str(datetime.now().minute)+'/fig_')
    cfg.Nf = 50 # was 50
    cfg.N_cpu = -1
    
    cfg.fu_alg = 'ls'
    cfg.mode = 'Relax'
    
    rob_rng = [0,1,2]
    snr_rng = np.hstack((np.linspace(-26,-22,3),np.linspace(-20,-10,11, dtype='int'),np.linspace(-8,10,10))) 
    Nsens_rng = [4,5,6,8,10,12]
    Nob_rng = np.linspace(1,31,16, dtype='int') 
    swidth_rng = [0.25,0.5,1,2,3,4,5,6,8]
    
    rob_std = 1
    sep_th_std = 1
    snr_std = -10
    Nsens_std=4
    Nob_std=10
    swidth_std = 4
    
    cfg.sep_th = sep_th_std

    ##################
    # Nob vs SNR
    # Nob_rng2 = [1,10, 20, 30]
    # set_it(1, snr_rng, [0,2,4],[rob_std, Nsens_std, swidth_std])
    # run_it(datef, Nob_rng2, 'snr','Nob')
    ##################
    # snr_rng2 = [-15, -10]
    # # # SNR vS Nob
    # set_it(3, Nob_rng, [0,2,4],[rob_std, Nsens_std, swidth_std])
    # run_it(datef, snr_rng2, 'Nob','snr')
    #################
    # Nsens_std2 = 6
    # rob_rng2 = [0, 1, 2]
    # # # Rob vs Nob 
    # set_it(3, Nob_rng, [1,2,4],[snr_std, Nsens_std, swidth_std])
    # run_it(datef, rob_rng2,'Nob','rob')
#     #################
    # Nsens_rng2 = [4,6]
    # # Nsens vs Nob
    # set_it(3, Nob_rng, [0,1,4],[rob_std, snr_std, swidth_std])
    # run_it(datef, Nsens_rng2,'Nob','Nsens')
#     ####################
    # Nob_rng2 = [5, 10, 15]
    # # Nob vS Nsens
    # set_it(2, Nsens_rng, [0,1,4],[rob_std, snr_std, swidth_std])
    # run_it(datef, Nob_rng2,'Nsens','Nob')
#     ################
# #    swidth_rng2 = [0.1, 0.2, 0.4, 0.8]
#     Nsens_std2= 6
#     # Rob vS swidth
#     set_it(4, swidth_rng, [1,3,2],[snr_std, Nob_std, Nsens_std2])
#     run_it(datef, np.arange(0,Nsens_std2-1),'swidth','rob')
#     ################
    Nsens_rng2 = np.array([5,6,7,8,9,10, 12])
    # Rob vS Nsens
    set_it(2, Nsens_rng2, [1,3,4],[snr_std, Nob_std, swidth_std])
    run_it(datef, np.arange(0,np.min(Nsens_rng)-1),'Nsens','rob')

#     ################
#     rob_rng2 = [0, 1, 2]
#     cfg.estalgo = 1
#     # DFT vS Nob
#     set_it(3, Nob_rng, [1,2,4],[snr_std, Nsens_std, swidth_std])
#     run_it(datef, rob_rng2,'Nob','rob')
#     ################
#     # DFT vS Nsens
#     set_it(2, Nsens_rng, [1,3,4],[snr_std, Nob_std, swidth_std])
#     run_it(datef, rob_rng2,'Nsens','rob')
#     ################
#     # DFT vS snr
#     set_it(1, snr_rng, [2,3,4],[Nsens_std, Nob_std, swidth_std])
#     run_it(datef, rob_rng2,'swidth','rob')

    
if __name__ == "__main__":
    __spec__ = None

    main()