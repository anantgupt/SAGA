#%%
"""
Kalman Filter
Implement single shot estimations for multi sensor localization
April 19, 2018
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import os
#mpl.use('TkAgg')
# import matplotlib.animation as manimation
from numpy import unravel_index
# import mpld3

# mpld3.enable_notebook()
# Add custom classes
from GAutils import objects as ob
from GAutils import proc_est as pr
from GAutils import PCRLB as pcrlb
from GAutils import ml_est as mle
from GAutils import gradient_methods as gm
from GAutils import perf_eval as prfe
from GAutils import config as cfg # Sim parameters
from GAutils import graph_primitives as grpr
from GAutils import est_algo as ea
from GAutils import mcft as mcft

# import importlib
# importlib.reload(cfg)
# def main():

def run_snapshot(scene, sensors, snr, cfgp, seed =int.from_bytes(os.urandom(4), byteorder='little')):
    tf_list = np.array([sensor.mcs.tf for sensor in sensors])  # All sensors frame times equal
    tfa_list = np.array([sensor.mcs.get_tfa() for sensor in sensors])  # Adjust so that samples vary to keep frame time const.

    beat = np.zeros(tfa_list.shape, dtype='complex128')
    dt = (1-int(cfgp['static_snapshot'])) *  tf_list[0] # make 0 to simulate one shot over Nf>1
    signal_mag =1 # TODO: Set this carefully
    for sensor in sensors:
        sensor.meas_std = 10 **(-snr/20)*signal_mag
    
    gardat = [ob.gardEst() for sensor in enumerate(sensors)]
    targets_list = []

    for tno, target in enumerate(scene):
        target_current, AbsPos = pr.ProcDyms(target, dt, tfa_list)# this adds noise to target state
        for sensorID, sensor in enumerate(sensors):
            random_number = np.random.rand()
            if random_number>cfgp['pmiss']: #Miss target otherwise
                pure_beat = pr.get_beat(sensor, target, AbsPos[sensorID])
                beat[sensorID, :, :] += pure_beat
            garda = pr.get_gard_true(sensor, target)
            gardat[sensorID].r=np.append(gardat[sensorID].r,garda.r)
            gardat[sensorID].d=np.append(gardat[sensorID].d,garda.d)
            gardat[sensorID].g=np.append(gardat[sensorID].g,garda.g)
        if not cfgp['static_snapshot']: targets_list.append(target_current)
    np.random.seed(seed) # To randomize over parallel runs
    for sensorID, sensor in enumerate(sensors):
        beat[sensorID, :, :] = pr.add_cnoise(beat[sensorID, :, :], sensor.meas_std) # Add noise
#        print('Target{}: x={},y={},vx={},vy={}'.format(tno, target_current.x, target_current.y,target_current.vx,target_current.vy))
    runtime = np.zeros(8)
    t=time.time()
    if cfgp['estalgo'] == 0:
        garda_sel = ea.meth2(np.copy(beat), sensors, cfgp['Nsel'], [1,1])
    elif cfgp['estalgo'] == 1:
        garda_sel = ea.meth2(np.copy(beat), sensors, cfgp['Nsel'], cfgp['osps'], cfgp['n_pfa'])
    elif cfgp['estalgo'] == 2:
        garda_sel = ea.nomp(np.copy(beat), sensors, cfgp['Nsel'], cfgp['osps'], cfgp['n_Rc'], cfgp['n_pfa'])
    runtime[0] = time.time() - t
    
    rd_error = prfe.compute_rd_error(garda_sel, gardat)
    rde_pack = prfe.compute_rde_targetwise(garda_sel, gardat, sensors)
    #%% Computer phantoms, tracks and llr's    
    rd_wt = cfgp['rd_wt'] # Range doppler relative weighting for likelihood, NLLS (Selection purposes)

     #%% Graph Algo
    G1,runtime[4] = grpr.make_graph(garda_sel, sensors, 0) # was cfgp['rob']
#        runtime[4] = sum([grpr.get_Ntracks(nd) for nd in G1[0]])# All tracks in graph
    runtime[4] = sum([len(nd.lkf) for g in G1 for nd in g]) # No of edges, get V from glen
    runtime[5],_ = grpr.get_BruteComplexity(G1)

    if cfgp['mode']=='mcf':
        min_gsigs, glen, runtime[6:8] = mcft.get_mcfsigs(garda_sel, sensors, cfgp)
    elif cfgp['mode']=='mcf_all':
        min_gsigs, glen, runtime[6:8] = mcft.get_mcfsigs_all(garda_sel, sensors, cfgp)
    elif cfgp['mode']=='mle':
        min_gsigs, glen, runtime[6:8] = mle.iterative_prune_pht(garda_sel, sensors, cfgp, sum(len(g.r) for g in garda_sel)//2)
    else:
        t=time.time()
        G0,runtime[4] = grpr.make_graph(garda_sel, sensors, 0) # No skip connection
        runtime[1] = time.time() - t
        if cfg.scene_plots:
            [graph_sigs, Ngsig]=grpr.enum_graph_sigs(G1, sensors)
            pr.plot_graph(G1, graph_sigs, sensors, rd_wt, 12, plt)
        min_gsigs, glen, runtime[6:8] = grpr.get_minpaths(G0, sensors, cfgp['mode'], cfgp)
    runtime[2] = time.time() - t # Total time (Make graph+traverse graph)

    t = time.time()
    for sig in min_gsigs:
        _,raw_ob = sig.get_rd_fit_error(sensors, cfgp['fu_alg'])
        [dob, nlls_var] = gm.gauss_newton(sig, sensors,[raw_ob.x,raw_ob.y,raw_ob.vx,raw_ob.vy] , cfgp['gn_steps'], rd_wt)#lm_refine, gauss_newton, huber
        sig.state_end.mean = dob
    runtime[3] = time.time() - t # Time to Refine

    gr_centers = []
    for gtr in min_gsigs:
        dob = gtr.state_end.mean
        gr_centers.append(ob.PointTarget(dob[0], dob[1], dob[2], dob[3]))
#    print ('{} detected {} targets in {}s.'.format(cfg.mode, len(min_gsigs),sum(runtime)))
    if cfg.scene_plots:
        plt.figure(13)
        for gtr in min_gsigs:
            dob = gtr.state_end.mean
            plt.quiver(dob[0], dob[1], dob[2], dob[3],color='b', headwidth = 4)
        pr.plot_scene(plt, scene, sensors, 13, 'Graph pruning detects {} targets'.format(len(min_gsigs)))
    #%% PLot likelihood maps
    if 0:
        [xgrid, ygrid, llr_map] = mle.create_llrmap([-9,9,180], [1,12,110], [-5,5,2], [-5,5,2], sensors, garda_sel) # Position
        cmap = plt.get_cmap('PiYG')
        plt.figure(16)
        im1 = plt.pcolormesh(xgrid, ygrid, llr_map, cmap=cmap)
        plt.colorbar(im1)
        pr.plot_scene(plt, scene, sensors, 3, 'Likelihood Map (Brute Force, Only using r)')
#%% Compute error measures
    ospa_error1, pv_error = prfe.compute_ospa(scene, gr_centers, sensors, gardat)
    if 1:# RD CRB
        [cr,cd, rList, dList]=pcrlb.get_FIMrv(sensors, scene)
    else:# RD ZZB
        cr = np.zeros((len(scene),len(sensors)))
        cd = np.zeros((len(scene),len(sensors)))
        for s, sensor in enumerate(sensors):
            for t, target in enumerate(scene):
                [cr[t,s], cd[t,s]] = pcrlb.ZZBrv(sensor, target)
# Convert to position, Vel bounds
    crb_conv = pcrlb.CRBconverter()
    [_,_,_,_,crbp, crbv] = crb_conv.get_CRBposvel_from_rd(cr, cd, sensors, scene)
#    [St_er[f,:], KF_er[f,:], Auto_er[f,:], sig_indx, sig_indx_auto, track_var, y_est_sig, vy_est_sig] = am.compute_asc_error(signatures, scene, Nsig, sensors) # compute error metrics
    results={'RDerror':np.array(rd_error),
             'RDpack':rde_pack,
    'OSPAerror1': ospa_error1,
    'runtime': runtime,
    'loc': gr_centers,
    'crbrd':np.stack([cr.T**2, cd.T**2],axis=-1),
    'crbpv': np.stack([np.array(crbp)**2, np.array(crbv)**2],axis=-1),
    'glen': glen,
    'garda': garda_sel,
    'PVerror': pv_error}
    results['next_scene'] = targets_list
    return results
        