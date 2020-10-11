#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:11:08 2019
Analyze performance of multi sensor localization algorithms
@author: anantgupta
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle
# from IPython import get_ipython
from functools import partial
import os as os
from tqdm import tqdm
import matplotlib.animation as animation
import time
# Custom libs
import GAutils.objects as ob
import GAutils.config as cfg # Sim parameters
import GAutils.proc_est as pr
import GAutils.simulate_snapshot2 as sim2
import GAutils.perf_eval as prfe
import GAutils.PCRLB as pcrlb

import importlib
importlib.reload(cfg)

def set_params(name, value):
    exec('cfg.'+name+' = value')
    
def main():
#if 1: # For spyder     
    Nsensa = cfg.Nsensa
    # Naming algorithm names & Plotting
    alg_name = ['Estimation', 'Graph Init.','Association','Refinement','All_edges','Brute',cfg.mode+'-Edges',cfg.mode+'-LLR']
    
    Nf = cfg.Nf
    Noba=cfg.Noba
    snra=cfg.snra

    static_snapshot = cfg.static_snapshot

    runtime = np.zeros([8,cfg.Ninst])

    ospa_error1 = np.zeros([cfg.Ninst,cfg.Nf,5])
    PVerror = np.zeros((cfg.Ninst, max(Noba),2))
    rd_error = np.zeros([cfg.Ninst,cfg.Nf,2])
    rd_err1 = np.zeros((cfg.Ninst, max(Nsensa), max(Noba),2))
    rd_err2 = np.zeros((cfg.Ninst, max(Nsensa), max(Noba),2))
    crb1 = np.zeros((cfg.Ninst, max(Nsensa), max(Noba),2))
    crbpv = np.zeros((cfg.Ninst, max(Noba),2))
    present = np.zeros((cfg.Ninst, max(Nsensa), max(Noba)))
    Nmiss1=np.zeros((cfg.Ninst, max(Nsensa)))
    Nfa1 =np.zeros((cfg.Ninst, max(Nsensa)))
    grca = [[] for _ in range(cfg.Ninst)]
    glena = np.zeros((cfg.Ninst, 100))
    Ndet = np.zeros((cfg.Ninst,cfg.Nf))
    plt.close('all')
    #for plt_n in range(1,6): plt.figure(plt_n), plt.clf()

    #%%
    # Arrange sensors in worst case to build up a scene
    sensorsa = []
    sx=np.linspace(-max(cfg.swidtha), max(cfg.swidtha), max(cfg.Nsensa))
    for x in sx:
        sensorsa.append(ob.Sensor(x,0))
        
    np.random.seed(29)
    seeda = np.random.randint(1000, size=Nf)
    # print('Seeds used:',seeda)
    # TODO NOTE: Min threshold might not be satisfied for all sensors!!
    scenea = [pr.init_random_scene(max(Noba), sensorsa, cfg.sep_th, seeda[f]) for f in range(Nf)]

    t=time.time()
    # Step 1: Init multiprocessing.Pool()
    if cfg.N_cpu <1:
        N_cpu = mp.cpu_count()
    else:
        N_cpu = cfg.N_cpu
    pool = mp.Pool(N_cpu)
    print('Using CPU count = ',str(N_cpu))
    # snap = partial(sim2.run_snapshot, )
    for inst in tqdm(range(cfg.Ninst), desc='Instances'):
        Nob = Noba[inst]
        Nsens = Nsensa[inst]
        swidth = cfg.swidtha[inst]
        # Generate sensor each time
        sx=np.linspace(-swidth/2, swidth/2,Nsens)
        sensors = [ob.Sensor(x,0) for x in sx]

        cfgp = {'Nsel': [],# Genie info on # targets
                'rd_wt':cfg.rd_wt,
                'static_snapshot': cfg.static_snapshot,
                'sep_th':cfg.sep_th,
                'pmiss':cfg.pmissa[inst],
                'estalgo':cfg.estalgo, 
                'osps':cfg.osps,
                'n_Rc':cfg.n_Rc,
                'n_pfa':cfg.n_pfa,
                # Association
                'rob':cfg.roba[inst],
                'mode': cfg.mode,
                'hscale':cfg.hscale,
                'incr':cfg.incr,
                'hN': cfg.hN,
                'ag_pfa':cfg.ag_pfa,
                'al_pfa':cfg.al_pfa,
                'Tlen':cfg.Tlen,
                # Gauss Newton
                'gn_steps':cfg.gn_steps,
                'fu_alg':cfg.fu_alg
                }
#        print('Running {} of {} '.format(inst+1, cfg.Ninst))
        if cfg.parallel:
            # snapshot_results = []
            argarray = [(scenea[f][0:Nob], sensors, snra[inst], cfgp, seeda[f]) for f in range(Nf)]
            snapshot_results = pool.starmap(sim2.run_snapshot, argarray)
        for f in tqdm(range(Nf),desc='Averaging', leave=False):  # Loop over frames
            if cfg.parallel:
                snapshot_result = snapshot_results[f]
            else:
                snapshot_result = sim2.run_snapshot(scenea[f][0:Nob], sensors, snra[inst], cfgp, seeda[f])
            
            Ndet[inst, f] = len(snapshot_result['loc']) # Count target associated
            runtime[:,inst] += snapshot_result['runtime']
            ospa_error1[inst,f,:] += snapshot_result['OSPAerror1'] # track based
            
            glen = snapshot_result['glen']
            glena[inst,:len(glen)] += np.array(glen)
            ret, det, Nmisst, Nfat, crbt, presentt = snapshot_result['RDpack']#prfe.compute_rde_targetwise(garda_sel, gardat, sensors)
            rd_error[inst,f,:] += np.sum(snapshot_result['RDerror'],axis =1) # Already Mutiplied by number of targets detected
            grca[inst].append( snapshot_result['loc'] )
            rd_err1[inst,:Nsens,:Nob,0] += np.array(ret)
            rd_err1[inst,:Nsens,:Nob,1] += np.array(det)
            rd_err2[inst,:Nsens,:Nob,0] += np.array(ret)**2
            rd_err2[inst,:Nsens,:Nob,1] += np.array(det)**2
            
            present[inst,:Nsens,:Nob] +=presentt
            crb1[inst,:Nsens,:Nob] += snapshot_result['crbrd']/Nf #crbt
            Nmiss1[inst,:Nsens] += Nmisst
            Nfa1[inst,:Nsens] += Nfat

            crbpv[inst,:Nob] += snapshot_result['crbpv']/Nf
            PVerror[inst,:Nob] += snapshot_result['PVerror']/Nf
#            for i in range(3,5):
#                print(grca[inst][0][i-3].x)
#                print(ospa_error1[inst,f,i])
            #Average or update scene
            if not static_snapshot: scene = snapshot_result['next_scene'] # Update scene for next timestep
    # Step 3: Don't forget to close
    pool.close()  

    print('Processing took {} s.'.format(time.time()-t))
    #%% Mask the arrays for averaging
    mask1 = np.ones((cfg.Ninst, max(Nsensa), max(Noba),2))
    for i in range(cfg.Ninst):
        mask1[i,:Nsensa[i],:Noba[i],:]=0
    rd_err1 = np.ma.array(rd_err1, mask=mask1)
    rd_err2 = np.ma.array(rd_err2, mask=mask1)
    crb1 = np.ma.array(crb1, mask=mask1)
    present = np.ma.array(present, mask=mask1[:,:,:,0])
    Nmiss1=np.ma.array(Nmiss1, mask=mask1[:,:,0,0])
    Nfa1 =np.ma.array(Nfa1, mask=mask1[:,:,0,0])
    crbpv = np.ma.array(crbpv, mask=mask1[:,0,:,:])
    PVerror = np.ma.array(PVerror, mask=mask1[:,0,:,:])
    #%% INterference CRB
    
    #%% Final Plotting
    # plt.switch_backend('Qt4Agg')  
    rng_used = cfg.rng_used
    units=['(m)','(m/s)']
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.bar(range(4), np.mean(runtime[:4], axis=1), tick_label=alg_name[:4]),plt.grid(True)
    plt.subplot(1,2,2)
    pltn={}
    for i in range(4):
        pltn[i]= plt.plot(rng_used, runtime[i,:], label = alg_name[i]),plt.grid(True)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(8.8,4.8)
    plt.tight_layout()
    # Track comparisons
    plt.figure(11)
    plt.subplot(1,2,1)
    plt.bar(range(3), np.mean(runtime[4:7], axis=1), tick_label=alg_name[4:7]),plt.grid(True)
    plt.ylabel('Number of Tracks visited'),plt.title('Association Complexity')
    plt.subplot(1,2,2)
    pltn={}
    for i in range(4,8):
        pltn[i]= plt.plot(rng_used, runtime[i,:], label = alg_name[i]),plt.grid(True)
    plt.legend(),plt.xlabel(cfg.xlbl),plt.ylabel('Number of Tracks visited'),plt.title('Association Complexity')
    plt.yscale('log')
    fig = plt.gcf()
    fig.set_size_inches(8.8,4.8)
    plt.tight_layout()
    # Analyze track quality
#    plt.figure(2)
#    plt.plot(St_er)
#    plt.xlabel(cfg.xlbl),plt.ylabel('RMS Error'),plt.title('Error Nearest Phantom(Solid), Auto KF(Dashed)')
#    plt.plot(Auto_er, linestyle='--'),plt.legend(['x','y','v_x','x','y','v_x'])
    # Ananlyze
    capt2 = ['Position error','Velocity error']
    plt.figure(2)
    for i in range(3,5):
        plt.subplot(1,2,i-2)
        # plt.errorbar(rng_used, np.mean(ospa_error1[:,:,i], axis=1), np.std(ospa_error1[:,:,i], axis=1), color='r')
        # plt.errorbar(rng_used, np.mean(np.sqrt(crbpv[:,:,i-3]), axis=(1)), np.std(np.sqrt(crbpv[:,:,i-3]), axis=(1)), color='k')
        # plt.plot(rng_used, 10*np.log10(np.mean(np.sqrt(PVerror[:,:,i-3]),axis=1)#/np.mean(Ndet,axis=1) #Original
        if True:
            # Find where are non zero PVerrors
            PVTemp = PVerror[:,:,i-3]
            CRBTemp = crbpv[:,:,i-3]
            plt.plot(rng_used, 10*np.log10([np.mean(np.sqrt(PVi[PVi>0])) for PVi in PVTemp]
                    ), color='r', label='RMSE')
            plt.plot(rng_used, 10*np.log10([np.mean(np.sqrt(CRBT[PVi>0])) for (PVi,CRBT) in zip(PVTemp,CRBTemp)]
                    ), 'k--', label='CRB'),plt.yscale('linear')
        else:
            plt.plot(rng_used, 10*np.log10(np.mean(np.sqrt(PVerror[:,:,i-3]),axis=1)
                    ), color='r', label='RMSE')
            plt.plot(rng_used, 10*np.log10(np.mean(np.sqrt(crbpv[:,:,i-3]),axis=1)
                    ), 'k--', label='CRB'),plt.yscale('linear')
        # plt.subplot(2,2,i)
        # for j in range(crbpv.shape[1]):
        #     plt.plot(rng_used, np.sqrt(PVerror[:,j,i-3]), color='r')
        #     plt.plot(rng_used, (np.sqrt(crbpv[:,j,i-3])), color='k'),plt.yscale('log')
        plt.xlabel(cfg.xlbl),plt.ylabel('RMS Error (dB)'+units[i-3]),plt.title(capt2[i-3]),plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(8,4.8)
    plt.tight_layout()
    capt3 = ['Overall','Localization error','Cardinality error']
    plt.figure(3)
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.errorbar(rng_used, np.mean(ospa_error1[:,:,i], axis=1), np.std(ospa_error1[:,:,i], axis=1), color='r')
        plt.xlabel(cfg.xlbl),plt.title(capt3[i]),plt.grid(True)
        if i<=1:
            plt.yscale('log'), plt.ylabel('RMS Error (?)')
        else:
            plt.ylabel('Error in Num targets')
    
    fig = plt.gcf()
    fig.set_size_inches(9.6,4.8)
    plt.tight_layout()
    capt4 = ['Range Error','Doppler Error']
    plt.figure(4)
    for i in range(2):
        plt.subplot(1,2,i+1)
        # plt.plot(rng_used, 10*np.log10(np.sum(np.sqrt(rd_err2[:,:,:,i]), axis =(1,2))/np.sum(present,axis=(1,2))), 'r-', label='RMSE')
        plt.plot(rng_used, 10*np.log10(np.sqrt(np.sum(rd_err2[:,:,:,i], axis =(1,2))/np.sum(present,axis=(1,2)))), 'r-', label='RMSE')
        plt.plot(rng_used, 10*np.log10(np.sqrt(np.mean(crb1[:,:,:,i], axis=(1,2)))), 'k--', label='CRB')
#        plt.plot(rng_used, 10*np.log10(np.mean(np.sqrt(crb1[:,:,:,i]), axis=(1,2))), 'k--', label='CRB')
        plt.xlabel(cfg.xlbl),plt.ylabel('RMS Error (dB)'+units[i]),plt.title(capt4[i]),plt.grid(True),plt.yscale('linear')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8,4.8)
    capt4 = ['Range Error, ','Doppler Error, ']
    if cfg.sensor_wise:
        plt.figure(5)
        for i in range(2):
            for j in range(Nsens):
                plt.subplot(2,Nsens, i*Nsens+j+1)
                plt.errorbar(rng_used, np.mean(rd_err1[:,j,:,i]/present[:,j,:],axis=1), 
                             np.sqrt(np.mean(rd_err2[:,j,:,i]/present[:,j,:]-(rd_err1[:,j,:,i]/present[:,j,:])**2, axis =1)),label='S{}'.format(j))
                if i==1: plt.xlabel(cfg.xlbl)
                if j==0: plt.ylabel('RMS Error '+units[i])
                plt.title(capt4[i]),plt.legend(),plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(12.8,7.2)
        plt.tight_layout()
        plt.figure(6)
        ax1, ax2  = plt.subplot(2,2,1), plt.subplot(2,2,2)
        for j in range(Nsens):
            ax1.plot(rng_used, np.mean(present[:,j,:],axis=1)/Nf, label='S{}'.format(j+1))
        ax1.set_title('Expected P(Detection), Miss, False Alarm'),ax1.set_xlabel(cfg.xlbl),ax1.grid(True),ax1.legend()
        for j in range(Nsens):
            tr_p = np.mean(present[:,j,:],axis=1)/Nf
            fa_p = Nfa1[:,j]/Nf
            fa_n = Nmiss1[:,j]/Nf
            precision_m = tr_p/(fa_p+tr_p)
            recall_m = tr_p/(tr_p+fa_n)
            ax2.scatter(recall_m, precision_m)
        ax2.set_title('Precision vs Recall'),ax2.set_ylabel('Precision'),ax2.set_xlabel('Recall'),ax2.grid(True)
        plt.subplot(2,2,3)
        for j in range(Nsens):
            plt.plot(rng_used, Nmiss1[:,j]/Nf, label='S{}'.format(j+1))
        plt.title('Missed targets'),plt.legend(),plt.grid(True),plt.xlabel(cfg.xlbl),plt.ylabel(r'$E\left[(N_{est}-N_{true})_-\right]$')
        plt.subplot(2,2,4)
        for j in range(Nsens):
            plt.plot(rng_used, Nfa1[:,j]/Nf, label='S{}'.format(j+1))
        plt.title('False Targets'),plt.legend(),plt.grid(True),plt.xlabel(cfg.xlbl),plt.ylabel(r'$E\left[(N_{est}-N_{true})_+\right]$')
        resizefig(plt, 8,6)
        plt.figure(8)
        for i in range(2):
            for j in range(Nsens):
                plt.subplot(2,Nsens,Nsens*i+j+1)
                for k in range(Nob):
                    plt.plot(rng_used, np.sqrt((rd_err2[:,j,k,i]/present[:,j,k]-(rd_err1[:,j,k,i]/present[:,j,k])**2)))
                plt.gca().set_prop_cycle(None)# Reset coloring
                for k in range(Nob):
                    plt.plot(rng_used, np.sqrt(crb1[:,j,k,i]/present[:,j,k]), '--')
                if i==1: plt.xlabel(cfg.xlbl)
                if j==0: plt.ylabel('RMS Error '+units[i])
                plt.title(capt4[i]+'Sensor '+str(j+1)),plt.grid(True),plt.yscale('log')
        resizefig(plt, 12.8,7.2)
    else:
        plt.figure(5)
        for i in range(2):
            plt.subplot(1,2, i+1)
            plt.errorbar(rng_used, np.mean(rd_err1[:,:,:,i]/present,axis=(1,2)), 
                         np.sqrt(np.mean(rd_err2[:,:,:,i]/present-(rd_err1[:,:,:,i]/present)**2, axis =(1,2))))
            plt.xlabel(cfg.xlbl),plt.ylabel('RMS Error'),plt.title(capt4[i]),plt.grid(True)
        plt.figure(6)
        plt.errorbar(rng_used, np.mean(present[:,:,:]/Nf, axis=(1,2)), np.std(present/Nf, axis=(1,2)),label='P_D')
        plt.errorbar(rng_used,np.mean( Nmiss1/Nf, axis=1),np.std( Nmiss1/Nf, axis=1), label= 'Miss')
        plt.errorbar(rng_used,np.mean( Nfa1/Nf, axis=1),np.std( Nfa1/Nf, axis=1),label = 'False Alarm')
        plt.title('Expected P(Detection), Miss, False Alarm'),plt.legend(),plt.grid(True),plt.xlabel(cfg.xlbl)
        plt.figure(8)
        for i in range(2):
            plt.subplot(1,2,i+1)
            plt.errorbar(rng_used, np.sqrt(np.mean(rd_err2[:,:,:,i]/present-(rd_err1[:,:,:,i]/present)**2, axis=(1,2))),
                             np.sqrt(np.std(rd_err2[:,:,:,i]/present-(rd_err1[:,:,:,i]/present)**2, axis=(1,2))))
            plt.errorbar(rng_used, np.sqrt(np.mean(crb1[:,:,:,i]/present,axis=(1,2))),
                 np.sqrt(np.std(crb1[:,:,:,i]/present,axis=(1,2))), fmt= '--')
            plt.gca().set_prop_cycle(None)# Reset coloring
            plt.xlabel('Sensor'),plt.ylabel('RMS Error'),plt.title(capt4[i]),plt.grid(True),plt.yscale('log')
    
    # plt.figure(7)
    fig, axs = plt.subplots(2, 2, num=7)# systemwide
    tr_p = np.array([ospa_error1[j,:,3]/Nob for j,Nob in enumerate(Noba)])
    fa_p = np.array([(ospa_error1[j,:,2]+Nob-ospa_error1[j,:,3])/Nob for j,Nob in enumerate(Noba)])
    fa_n = np.array([(Nob-ospa_error1[j,:,3])/Nob for j,Nob in enumerate(Noba)])
    precision_m = tr_p/(fa_p+tr_p)
    recall_m = tr_p/(tr_p+fa_n)
    axs[0,0].errorbar(rng_used, np.mean(tr_p,axis=1),np.std(tr_p,axis=1), label='P_D')
    axs[0,0].errorbar(rng_used, np.mean(fa_p,axis=1),np.std(fa_p,axis=1), label = 'False Alarm')
    axs[0,0].errorbar(rng_used, np.mean(fa_n,axis=1),np.std(fa_n,axis=1), label = 'Miss')
    axs[0,0].set_title('Expected P(Detection), Miss, False Alarm'),axs[0,0].set_ylabel(r'$P_D$')
    axs[0,0].set_xlabel(cfg.xlbl),axs[0,0].grid(True),axs[0,0].legend()
    axs[0,1].scatter(recall_m, precision_m)
    axs[0,1].set_title('Precision vs Recall'),axs[0,1].set_ylabel('Precision'),axs[0,1].set_xlabel('Recall'),axs[0,1].grid(True)
    axs[1,0].hist([Nob + ospa_error1[j,:,2] for j,Nob in enumerate(Noba)])
    axs[1,0].set_title('Histogram of detections (system-level)')
    resizefig(plt, 8,6)
    # Add plot for combined measure (P(estimate in ball|detect))
    plt.figure(9)
    for j in range(Nsens):
        plt.subplot(2,Nsens,j+1)
        prfe.plotg(rd_err1[:,j,:,0].flatten(), np.sqrt(np.sum(crb1[:,j,:,0],
                   axis=(0,1))/sum(Noba*Nsens)),plt,True),plt.title(r'$\Delta R$ Sensor {}'.format(j+1))
        plt.subplot(2,Nsens,Nsens+j+1)
        prfe.plotg(rd_err1[:,j,:,1].flatten(), np.sqrt(np.sum(crb1[:,j,:,1],
                   axis=(0,1))/sum(Noba*Nsens)),plt,True),plt.title(r'$\Delta D$ Sensor {}'.format(j+1))
    fig = plt.gcf()
    fig.set_size_inches(12.8,7.2)
    plt.tight_layout()
    plt.figure(10)
    plt.subplot(1,2,1)
    for i in range(cfg.Ninst):
        hN_max = np.count_nonzero(glena[i,:])
        plt.plot(range(hN_max+2), (glena[i,:hN_max+2]/Nf), label = str(rng_used[i]))
    plt.legend(),plt.grid(True),plt.title('Graph nodes v/s relax iterations'),plt.ylabel('Num vertices'),plt.xlabel('Iterations')
    plt.subplot(1,2,2)
    plt.errorbar(rng_used, np.mean(Ndet, axis=1), np.std(Ndet, axis =1), label = 'Estimated')
    plt.plot(rng_used, cfg.Noba, 'k:', label = 'True')
    plt.legend(),plt.grid(True),plt.title('Model order estimation'),plt.ylabel('Num targets detected'),plt.xlabel(cfg.xlbl)
    resizefig(plt, 8,4.8)
    # Save files
    try:
        # Create target Directory
        os.makedirs(cfg.folder)
        print("Directory " , cfg.folder ,  " Created ") 
    except FileExistsError:
        print("Directory " , cfg.folder ,  " already exists")
    # Setup video files
    if cfg.movie:
        try:
            FFMpegWriter = animation.writers['ffmpeg']
            metadata = dict(title='Movie Test', artist='Anant',comment='Target motion')
            writer = FFMpegWriter(fps=1, metadata=metadata)
            fig = plt.figure(15)
            with writer.saving(fig, '{}/Scenes.mp4'.format(cfg.folder), dpi=100):
                for i, scene in enumerate(scenea):
                    for j in range(cfg.Ninst):
                        sx=np.linspace(-cfg.swidtha[j], cfg.swidtha[j],cfg.Nsensa[j])
                        sensorsp = [ob.Sensor(x,0) for x in sx]
                        phlist = grca[j][i]
                        plt.clf()
                        for gr in phlist: 
                            if abs(gr.vx)+abs(gr.vy)>0:
                                plt.quiver(gr.x, gr.y,gr.vx,gr.vy, color='r', headwidth = 4, headlength=6, headaxislength=5)
                            else:
                                plt.plot(gr.x, gr.y, 'ro')
                        pr.plot_scene(plt, scene[:Noba[j]], sensorsp, 15, 'Scene {} with {} detections, SNR = {} dB'.format(i, np.round(np.sum(present[j,:,:],axis=1)/Nf/Noba[j],2), round(snra[j])))
                        writer.grab_frame()
        except Exception as e: print(e)
    
    # Save variables 
    # np.savetxt('{}/mat.out'.format(cfg.folder), (Noba, snra), delimiter=",")
    handle = open('{}/params.txt'.format(cfg.folder),'w') 
    
    handle.write('Robust Level={}\n'.format(cfg.roba))
    handle.write('Sep_th={}\n'.format(cfg.sep_th))
    handle.write('SNR={}\n'.format(np.round(snra,2))) 
    handle.write('Nsens={}\n'.format(cfg.Nsensa))
    handle.write('Noba={}\n'.format(np.round(Noba,2)))
    handle.write('Sensor Width={}\n'.format(cfg.swidtha))
    mcss=sensors[0].mcs
    handle.write('Sensor BW={}Hz,R_res={}m, D_res={}m/s \n'.format(mcss.B, 3e8/2/mcss.B, 3e8/2/mcss.fc/mcss.tf))
    handle.write('Monte Carlo Iterations={}\n'.format(cfg.Nf))

    handle.write('mode={}\n'.format(cfg.mode))
    handle.write('Tlen={}\n'.format(cfg.Tlen))
    handle.write('Pmiss={}\n'.format(cfg.pmissa))
    handle.write('Est_Algo={}\n'.format(cfg.estalgo))
    handle.write('NOMP: OSPS={}, n_pfa={}, n_Rc={}\n'.format(cfg.osps,cfg.n_pfa,cfg.n_Rc))
    handle.write('GA-DFS: ag_pfa={}, al_pfa={}\n'.format(cfg.ag_pfa, cfg.al_pfa))
    handle.write('Relax: hN={}, hscale={}, incr ={}\n'.format(cfg.hN, cfg.hscale, cfg.incr))
    handle.write('Misc: rd_wt={}, fu_alg={}, gn_steps={}'.format(cfg.rd_wt, cfg.fu_alg, cfg.gn_steps))

    for fignum in range(1,12):
        plt.figure(fignum)
        plt.savefig("{}/{}".format(cfg.folder,fignum), Transparent=True)
        if fignum not in [5,8,9]:
            pickle.dump(plt.figure(fignum), open("{}/plot{}.pickle".format(cfg.folder,fignum), "wb"))
    plt.close('all')
    print('Processing+Plotting took {} s.'.format(time.time()-t))


def resizefig(plt, x, y):
    fig = plt.gcf()
    fig.set_size_inches(x,y)
    plt.tight_layout()
        
if __name__ == "__main__":
    __spec__ = None
#    ipython = get_ipython()
#    ipython.magic('%load_ext autoreload')
#    ipython.magic('%autoreload 2')    
#    ipython.magic('%matplotlib')
    main()
