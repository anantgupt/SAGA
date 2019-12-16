#%% [markdown]
# # Simulate localization project

#%%
#%% Import stuff
"""
Kalman Filter
Implement Kalman filter in Python
April 19, 2018
"""
import numpy as np
import time
import matplotlib as mpl # To draw movie
import matplotlib.pyplot as plt
#mpl.use('TkAgg')
# import matplotlib.animation as manimation
from numpy import unravel_index
import pickle
import pandas as pd
import seaborn as sns
import importlib
import copy as cp
#from plotly.offline import init_notebook_mode, iplot
#import plotly.graph_objs as go
# import mpld3

# mpld3.enable_notebook()
# Add custom classes
from GAutils import objects as ob
from GAutils import proc_est as pr
from GAutils import association_methods as am
from GAutils import PCRLB as pcrlb
from GAutils import ml_est as mle
from GAutils import gradient_methods as gm
from GAutils import perf_eval as prfe
from GAutils import config as cfg # Sim parameters
from GAutils import bel_prop as bp
from GAutils import graph_primitives as grpr
from GAutils import est_algo as ea
#from GAutils import iter_prune as itpr

#init_notebook_mode()
np.set_printoptions(precision=2)# REduce decimal digits
#%% [markdown]
# Place $N_s$ sensors and create $N_t$ targets 

#%%
# def main():
scene_init = []  # Scene: List of targets at a time
# scenes = [[] for _ in Nf]  # List of scenes across frames
scene_init.append(ob.PointTarget(2, 5, 1, -2, 0.1, 1))
scene_init.append(ob.PointTarget(1, 2, -5, -2, 0.1, 1))
scene = scene_init
signal_mag =1 # NOTE: Set this carefully
#    targets[0]=[PointTarget(x,y,1) for x,y in np.random(2,Nob)*10]
#sensors = []
#sensors.append(ob.Sensor(-5, 0))
#sensors.append(ob.Sensor(-3, 0))
#sensors.append(ob.Sensor(-1, 0))
#sensors.append(ob.Sensor(1, 0))
#sensors.append(ob.Sensor(3, 0))
#sensors.append(ob.Sensor(5, 0))
sensors = [ob.Sensor(x,0) for x in np.linspace(-2,2,9)]

tf_list = np.array([sensor.mcs.tf for sensor in sensors])  # All sensors frame times equal
tfa_list = np.array([sensor.mcs.get_tfa() for sensor in sensors])  # Adjust so that samples vary to keep frame time const.
Nf = 1 #cfg.Nf
Noba = [25] #cfg.Noba
static_snapshot = 1

## Estimation Parameters
rd_wt = cfg.rd_wt # Range doppler relative weighting for likelihood, NLLS (Selection purposes)

alg_name = ['DFT2', 'Interp DFT2','NOMP']
colr=['r','b','g']
runtime = np.zeros([3,Nf])
rtime_algo = dict()
# snra = np.linspace(-20,10,Nf)
snra = np.ones(Nf)*-15
# Setup video files
#plot_scene(fig, scene_init, sensors, 3)
# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Anant',comment='Target motion')
# writer = FFMpegWriter(fps=2, metadata=metadata)
St_er = np.zeros([Nf,3])
Auto_er = np.zeros([Nf,3])
KF_er = np.zeros([Nf,2])
asc_targets = np.zeros(Nf)
ospa_error = np.zeros([Nf,3])
plt.close('all')

np.random.seed(5)

cfgp = {'Nsel': [],# Genie info on # targets
                'rd_wt':cfg.rd_wt,
                'static_snapshot': cfg.static_snapshot,
                'sep_th':cfg.sep_th,
                'pmiss':cfg.pmiss,
                'estalgo':cfg.estalgo, 
                'osps':cfg.osps,
                'n_Rc':cfg.n_Rc,
                'n_pfa':cfg.n_pfa,
                # Association
                'rob':cfg.roba[0],
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
cfgp['rob'] = 2
cfgp['pmiss']=0.15
cfgp['mode']='Relax' # SPEKF, Relax

for f in range(Nf):  # Loop over frames
    targets_list = []
    for plt_n in range(3,6): plt.figure(plt_n), plt.clf()
    beat = np.zeros(tfa_list.shape, dtype='complex128')
    dt = static_snapshot *  tf_list[0] # make 0 to simulate one shot over Nf>1
    Nob = Noba[f]
    Nsel = Nob
    scene = pr.init_random_scene(Nob, sensors, 0)
    for sensor in sensors:
        sensor.meas_std = 10 **(-snra[f]/20)*signal_mag
    
    gardat = [ob.gardEst() for sensor in enumerate(sensors)]
    for tno, target in enumerate(scene):
        target_current, AbsPos = pr.ProcDyms(target, dt, tfa_list)
        for sensorID, sensor in enumerate(sensors):
            random_number = np.random.rand()
            if random_number>cfgp['pmiss']: #Miss target otherwise
                pure_beat = pr.get_beat(sensor, target, AbsPos[sensorID])
                beat[sensorID, :, :] += pure_beat
            garda = pr.get_gard_true(sensor, target)
            gardat[sensorID].r=np.append(gardat[sensorID].r,garda.r)
            gardat[sensorID].d=np.append(gardat[sensorID].d,garda.d)
            gardat[sensorID].g=np.append(gardat[sensorID].g,garda.g)
        if not static_snapshot: targets_list.append(target_current)
#        print('Target{}: x={},y={},vx={},vy={}'.format(tno+1, target_current.x, target_current.y,target_current.vx,target_current.vy))
    for sensorID, sensor in enumerate(sensors):
        beat[sensorID, :, :] = pr.add_cnoise(beat[sensorID, :, :], sensor.meas_std) # Add noise
    t=time.time()
    garda1 = ea.meth2(np.copy(beat), sensors, Nob, [1,1])
    runtime[0,f] = time.time() - t
    t = time.time()
    garda2 = ea.meth2(np.copy(beat), sensors, Nob, cfg.osps)
    runtime[1,f] = time.time() - t
    t= time.time()
    garda3 = ea.nomp(np.copy(beat), sensors)
    runtime[2,f] = time.time() - t
    #        plotdata(targets[0].x,targets[0].y,1)
    garda_sel = garda3
    plt.figure(27)
    rd_error = np.array(prfe.compute_rd_error(garda_sel, gardat, plt))
    ##
#    if cfg.all_pht: pht_all = am.associate_garda(garda_sel, sensors)# Use all pairs of phantoms 
#    else: pht_all = am.associate_garda2(garda_sel, sensors)# Use pairwise or all phantoms 
#    N_pht = len(pht_all)
#    print ('Phantoms:',N_pht)
#    ordered_links, forward_links = am.band_prune(garda_sel, sensors)
#    pr.plot_orderedlinks(ordered_links, garda_sel, sensors, rd_wt, 21, plt)
    #%% Graph Algo
    
    rob = cfgp['rob']
    t=time.time()
    G1, rtime_make = grpr.make_graph(garda_sel, sensors, 0)
    print('Graph gen took {}s'.format(rtime_make))
    G0 = cp.deepcopy(G1)
    if False:    
        [graph_sigs, Ngsig]=grpr.enum_graph_sigs(G0, sensors)
        pr.plot_graph(G1, graph_sigs, sensors, rd_wt, 78, plt, garda_sel) # All edges
    crb_min =np.array([1e-2, 1e-2])
    t=time.time()
    min_gsigs1, glen, rtime_assoc = grpr.get_minpaths(G0, sensors, cfgp['mode'], cfgp)
    print('{} Association took {}, {}s'.format(cfgp['mode'], rtime_assoc, time.time()-t))
        #%%
    pr.plot_graph(G1, min_gsigs1, sensors, rd_wt, 77, plt, garda_sel) # From Relax
    for sig in min_gsigs1:
        [new_pos, nlls_var] = gm.gauss_newton(sig, sensors, sig.state_end.mean , 5, rd_wt)
        sig.state_end.mean = new_pos
    rtime_algo["Graph"]= time.time()-t
    plt.figure(76)
    for gtr in min_gsigs1:
        dob = gtr.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='r')
#        print(dob, gtr.r)
    pr.plot_scene(plt, scene, sensors, 76, '{} detects {} targets'.format(cfgp['mode'], len(min_gsigs1)))
   
    #%% MCF Association
    cfgp['mode']='Relax-heap' # SPEKF, Relax, Rel3, Relax4
    from GAutils import mcft as mcft
    t=time.time()
    min_gsigs3, glen3, rtime_assoc3 = grpr.get_minpaths(cp.deepcopy(G1), sensors, cfgp['mode'], cfgp) # mcft.get_mcfsigs(garda_sel, sensors)
#    min_gsigs3, glen3, rtime_assoc3 = mcft.get_mcfsigs(garda_sel, sensors, cfgp)
#    min_gsigs3, glen3, rtime_assoc3 = mcft.get_mcfsigs_all(garda_sel, sensors, cfgp)
    print('{} Association took {}, {}s'.format(cfgp['mode'], rtime_assoc3, time.time()-t))
    pr.plot_graph(G1, min_gsigs3, sensors, rd_wt, 79, plt, garda_sel) # From Relax
    #%%
#    print('--')
    for sig in min_gsigs3:
        [new_pos, nlls_var] = gm.gauss_newton(sig, sensors, sig.state_end.mean , 5, rd_wt)
        sig.state_end.mean = new_pos
    plt.figure(75)
    for gtr in min_gsigs3:
        dob = gtr.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='r')
#        print(dob, gtr.r)
    pr.plot_scene(plt, scene, sensors, 75, '{} detects {} targets'.format(cfgp['mode'], len(min_gsigs3)))
    #%% ML est
    t=time.time()
    min_gsigs4, glen4, rtime_assoc4 = mle.iterative_prune_pht(cp.deepcopy(garda_sel), sensors, cfgp, Nob)
    print('{} Association took {}, {}s'.format('ML', rtime_assoc4, time.time()-t))
    for sig in min_gsigs4:
        [new_pos, nlls_var] = gm.gauss_newton(sig, sensors, sig.state_end.mean , 5, rd_wt)
        sig.state_end.mean = new_pos
    plt.figure(87)
    for ctr in centers:
        dob = ctr.state
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='g')
    for gtr in min_gsigs4:
        dob = gtr.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='r')
    pr.plot_scene(plt, scene, sensors, 87, '{} detects {} targets'.format('ML', len(min_gsigs4)))
    #%%
    break # Stop here (Older code ahead)
# PLot Glen 2D:
fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(glen, label = 'Relax chain length first')
ax.plot(glen3, label = 'Relax chain cost first')
ax.legend(loc='best'),ax.grid(True),ax.set_xlabel('Iterations'),ax.set_ylabel(r'Num nodes in Graph $\mathscr{G}$')