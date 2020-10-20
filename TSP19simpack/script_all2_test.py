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
import argparse

def set_it(itrx, xval, idxa, val):
    it_name=['roba','snra','Nsensa','Noba','swidtha','pmissa']
    it_xlbl = ['Robust level','SNR (dB)','Num. Sensors','Num. Targets','Array width (m)','Pmiss']
    exec("ms3.set_params('"+it_name[itrx]+"',xval)")
    ms3.set_params('Ninst',len(xval))
    ms3.set_params('rng_used',xval)
    ms3.set_params('xlbl',it_xlbl[itrx])
    for i, idx in enumerate(idxa):
        exec("ms3.set_params('"+it_name[idx]+"',np.ones(len(xval), dtype='int')*val[i])")

def run_it(datef, rng, itrx, itry):
    for num, i in enumerate(rng):
        exec("cfg."+itry+"a=i*np.ones(cfg.Ninst, dtype='int')")
        cfg.folder= datef +itrx+'-'+itry+str(i).replace('.','_')
        print('Running '+str(num)+'/'+str(len(rng)))
        ms3.main()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='SAGA',type=str, help='Association algorithm')
    parser.add_argument('--fu_alg', default='ls',type=str, help='Refinement algorithm')
    parser.add_argument('--sep_th', default=0, type=float, help='Separation threshold')
    parser.add_argument('--rob', default=20, type=int, help='Default Robustness level')
    parser.add_argument('--N_cpu', default=-1, type=int, help='CPU Count')
    parser.add_argument('--N_avg', default=50, type=int, help='Monte-Carlo iterations')
    parser.add_argument('--pmiss', default=0.05, type=float, help='Miss probability')
    parser.add_argument('--save_dir', default='paper_plots', type=str, help='Results save directory')

    args = parser.parse_args()

    datef =('results'+str(date.today().month)+str(date.today().day)
        +'_'+str(datetime.now().hour)+str(datetime.now().minute))
        # +str(np.random.randint(100))+args.mode+'/fig_')
    save_dir = args.save_dir
    cfg.Nf = args.N_avg # was 50
    cfg.N_cpu = (args.N_cpu)
    
    cfg.fu_alg = args.fu_alg # 'ls'
    cfg.mode = args.mode #'Relax'
    
    rob_rng = [0,1,2, 20]
    snr_rng = np.hstack((np.linspace(-26,-22,3),np.linspace(-20,-10,11, dtype='int'),np.linspace(-8,10,10))) 
    Nsens_rng = [4,5,6,8,10,12]
    Nob_rng = np.linspace(1,31,4, dtype='int') 
    swidth_rng = [0.25,0.5,1,2,4,6,8]
    
    rob_std = (args.rob)
    snr_std = -10
    Nsens_std= 6
    Nob_std=20
    swidth_std = 4
    pmiss_std = 0
    pmiss_std2 = 0.2
    pmiss_std3 = 0.05
    mode_dic = ['SAGA','SAESL', 'NN', 'MCF']

    cfg.sep_th = args.sep_th

    # ========== EXPERIMENTAL ===============
    # cfg.n_Rc = [1, 2] # default [1, 2]
    cfg.parallel = True
    # cfg.estalgo = 2
    ##################
    '''# Nob vs SNR
    Nob_rng2 = [1,10, 20, 30]
    set_it(1, snr_rng, [0,2,4,5],[rob_std, Nsens_std, swidth_std, pmiss_std])
    run_it(save_dir+'/Fig3_PVRD/fig_', Nob_rng2, 'snr','Nob')'''
    ##################
    for i in range(1):
        if (i==1):
            cfg.mode = mode_dic[i]
        snr_rng2 = [0] #[-15, -10, 0]
        # # SNR vS Nob
        set_it(3, Nob_rng, [0,2,4,5],[rob_std, Nsens_std, swidth_std, pmiss_std])
        run_it(save_dir+'/Fig4_OSPA/'+mode_dic[i]+'/fig_', snr_rng2, 'Nob','snr')
        #################
        pmiss_rng = np.linspace(0,0.8,9)
        '''# Rob vs Pmiss
        set_it(5, pmiss_rng, [1,3,4,2],[snr_std, Nob_std, swidth_std,Nsens_std])
        run_it(save_dir+'/Fig5_Pmiss/'+mode_dic[i]+'/fig_',[0,1,2,4,8],'pmiss','rob') 
        #################
        rob_rng2 = [0,1,2,20]
        pmiss_std2 = 0.05
        # # Rob vs Nob 
        set_it(3, Nob_rng, [1,2,4,5],[snr_std, Nsens_std, swidth_std, pmiss_std2])
        run_it(save_dir+'/Fig6_Complexity/'+mode_dic[i]+'/fig_', rob_rng2,'Nob','rob')
        # ###############
        Nsens_rng2 = np.array([4,5,6,7,8,10,12])
        # Rob vS Nsens
        set_it(2, Nsens_rng2, [1,3,4,5],[snr_std, Nob_std, swidth_std,pmiss_std3])
        run_it(save_dir+'/Fig11_12_Nsens/'+mode_dic[i]+'/fig_', [0,1,2,4,7,20],'Nsens','rob')'''

    '''# # # Pmiss vs Nob 
    for i in range(4):
        cfg.mode = mode_dic[i]
        pmiss_rng2 = [0,0.05,0.2,0.4]
        set_it(3, Nob_rng, [0,1,2,4],[rob_std, snr_std, Nsens_std, swidth_std])
        run_it(save_dir+'/Fig7_8_SOTA/'+mode_dic[i]+'/fig_', pmiss_rng2,'Nob','pmiss')'''

#     ####################
    '''cfg.mode = mode_dic[0]
    Nob_rng2 = [10, 20]
    Nsens_rng3 = [4,6,8,10,12,16]
    # Nob vS Nsens
    set_it(2, Nsens_rng3, [0,1,4,5],[rob_std, snr_std, swidth_std,pmiss_std3])
    run_it(save_dir+'/Fig10_DFT/fig_', Nob_rng2,'Nsens','Nob')'''
#     ################
   
   
    # ################
    # ## DFT
    # ################
    '''cfg.estalgo = 1
    # SNR vS Nob
    set_it(3, Nob_rng, [0,2,4,5],[rob_std, Nsens_std, swidth_std,pmiss_std])
    run_it(save_dir+'/Fig9_DFT/fig_DFT_', snr_rng2,'Nob','snr')'''
    '''
    ################
    Nob_rng2 = [10, 20]
    Nsens_rng3 = [4,6,8,10,12,16]
    # Nob vS Nsens
    set_it(2, Nsens_rng3, [0,1,4,5],[rob_std, snr_std, swidth_std,pmiss_std3])
    run_it(save_dir+'/Fig10_DFT/fig_DFT_', Nob_rng2,'Nsens','Nob')'''
    
if __name__ == "__main__":
    __spec__ = None

    main()