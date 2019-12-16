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
import cProfile
import copy as cp
from IPython import get_ipython

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
from GAutils import mcft as mcft

#init_notebook_mode()
np.set_printoptions(precision=6)# REduce decimal digits
#%% [markdown]
# Place $N_s$ sensors and create $N_t$ targets 
ipython = get_ipython()
ipython.magic('%load_ext autoreload')
ipython.magic('%autoreload 2')    
ipython.magic('%matplotlib')
#%%
# def main():
scene_init = []  # Scene: List of targets at a time
# scenes = [[] for _ in Nf]  # List of scenes across frames
scene_init.append(ob.PointTarget(3, 4, 0, 0, 0.1, 1))
scene_init.append(ob.PointTarget(-0.25, 6.11, 0, 0, 0.1, 1))
scene = scene_init
signal_mag =1 # NOTE: Set this carefully
#    targets[0]=[PointTarget(x,y,1) for x,y in np.random(2,Nob)*10]
sensors = []
#sensors.append(ob.Sensor(-5, 0))
sensors.append(ob.Sensor(-3, 0))
sensors.append(ob.Sensor(-1, 0))
sensors.append(ob.Sensor(1, 0))
sensors.append(ob.Sensor(3, 0))
#sensors.append(ob.Sensor(5, 0))
Nsens = 4

tf_list = np.array([sensor.mcs.tf for sensor in sensors])  # All sensors frame times equal
tfa_list = np.array([sensor.mcs.get_tfa() for sensor in sensors])  # Adjust so that samples vary to keep frame time const.
Nf = 1 # cfg.Nf
Noba = [2] #cfg.Noba
static_snapshot = 1

## Estimation Parameters
rd_wt = cfg.rd_wt # Range doppler relative weighting for likelihood, NLLS (Selection purposes)

alg_name = ['DFT2', 'Interp DFT2','NOMP']
colr=['r','b','g']
runtime = np.zeros([3,Nf])
rtime_algo = dict()
# snra = np.linspace(-20,10,Nf)
snra = np.ones(Nf)*(10)
# Setup video files
#plot_scene(fig, scene_init, sensors, 3)
# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Anant',comment='Target motion')
# writer = FFMpegWriter(fps=2, metadata=metadata)
St_er = np.zeros([Nf,3])
Auto_er = np.zeros([Nf,3])
KF_er = np.zeros([Nf,2])
asc_targets = np.zeros(Nf)
ospa_error = np.zeros([Nf,5])
rd_err1 = np.zeros((Nsens, Noba[0],2))
rd_err2 = np.zeros((Nsens, Noba[0],2))
rd_errall = np.zeros((Nf,Nsens, Noba[0],2))
crb1 = np.zeros((Nsens, Noba[0],2))
present = np.zeros((Nsens, Noba[0]))
Nmiss1=np.zeros(Nsens)
Nfa1 =np.zeros(Nsens)
plt.close('all')
#scene = pr.init_random_scene(Noba[0], sensors, 1, 257) # Static scene
for f in range(Nf):  # Loop over frames
    targets_list = []
    for plt_n in range(11,18): plt.figure(plt_n), plt.clf()
    beat = np.zeros(tfa_list.shape, dtype='complex128')
    dt = (1-static_snapshot) *  tf_list[0] # make 0 to simulate one shot over Nf>1
    Nob = Noba[0]
    Nsel = Nob
    
    for sensor in sensors:
        sensor.meas_std = 10 **(-snra[f]/20)*signal_mag
    
    gardat = [ob.gardEst() for sensor in enumerate(sensors)]
    for tno, target in enumerate(scene):
        target_current, AbsPos = pr.ProcDyms(target, dt, tfa_list)# this adds noise to target state
        for sensorID, sensor in enumerate(sensors):
            pure_beat = pr.get_beat(sensor, target, AbsPos[sensorID])
            beat[sensorID, :, :] += pure_beat
            garda = pr.get_gard_true(sensor, target)
            gardat[sensorID].r=np.append(gardat[sensorID].r,garda.r)
            gardat[sensorID].d=np.append(gardat[sensorID].d,garda.d)
            gardat[sensorID].g=np.append(gardat[sensorID].g,garda.g)
        if not static_snapshot: targets_list.append(target_current)
            
    for sensorID, sensor in enumerate(sensors):
        beat[sensorID, :, :] = pr.add_cnoise(beat[sensorID, :, :], sensor.meas_std) # Add noise
        

#        print('Target{}: x={},y={},vx={},vy={}'.format(tno+1, target_current.x, target_current.y,target_current.vx,target_current.vy))
    estalgo=2
    t=time.time()
    if estalgo==0:
        garda_sel = ea.meth2(np.copy(beat), sensors, Nob, [1,1])
    elif estalgo==1:
        garda_sel = ea.meth2(np.copy(beat), sensors, Nob, cfg.osps)
    elif estalgo==2:
    #    cProfile.run('garda3 = pr.nomp(np.copy(beat), sensors, [], [2,2],[1,3])')
        garda_sel = ea.nomp(np.copy(beat), sensors, [], cfg.osps, cfg.n_Rc, cfg.n_pfa)
    runtime[estalgo,f] = time.time() - t
    print(time.time()-t)
    #        plotdata(targets[0].x,targets[0].y,1)
#    plt.figure(27)
#    rd_error = np.array(prfe.compute_rd_error(garda_sel, gardat, plt))
    ## Compute bias, variance in RD estimation
    ret, det, Nmisst, Nfat, crbt, presentt = prfe.compute_rde_targetwise(garda_sel, gardat, sensors)
    rd_err1[:,:,0] += np.array(ret)
    rd_err1[:,:,1] += np.array(det)
    rd_err2[:,:,0] += np.array(ret)**2
    rd_err2[:,:,1] += np.array(det)**2
    rd_errall[f, :,:,0] =  np.array(ret)
    rd_errall[f, :,:,1] =  np.array(det)
    
    present+=presentt
    crb1 += crbt
    Nmiss1 += Nmisst
    Nfa1 += Nfat
    
    ## Only for plotting complete graph initialization with likelihood based edge coloring
    if 0:
        if cfg.all_pht: pht_all = am.associate_garda(garda_sel, sensors)# Use all pairs of phantoms 
        else: pht_all = am.associate_garda2(garda_sel, sensors)# Use pairwise or all phantoms 
        N_pht = len(pht_all)
        print ('Phantoms:',N_pht)
        ordered_links, forward_links = am.band_prune(garda_sel, sensors)
        pr.plot_orderedlinks(ordered_links, garda_sel, sensors, rd_wt, 21, plt)
    #%% Graph Algo
    cfgp = {'Nsel': [],# Genie info on # targets
                'rd_wt':cfg.rd_wt,
                'static_snapshot': cfg.static_snapshot,
                'sep_th':cfg.sep_th,
                'pmiss':cfg.pmissa[0],
                'estalgo':cfg.estalgo, 
                'osps':cfg.osps,
                'n_Rc':cfg.n_Rc,
                'n_pfa':cfg.n_pfa,
                # Association
                'rob':int(cfg.roba[0]),
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
    
    cfgp['mode'] = 'Relax' # Choose:'Brute','DFS','Relax','Brute_iter'
    crb_min =np.array([1e-2, 1e-2])
    t=time.time()
    
    print(time.time()-t)
    t=time.time()
    if 2: #cfgp['mode'] == 'mcf':
        min_gsigs2, glen,_ = mcft.get_mcfsigs(garda_sel,sensors, cfgp)
        glen = [1,1]
    if 1:
        G2,_ = grpr.make_graph(garda_sel, sensors, True)
        min_gsigs, glen,_ = grpr.get_minpaths(G2, sensors, cfgp['mode'], cfgp)
    rtime_algo[cfgp['mode']]= time.time()-t
    print(time.time()-t)
    plt.figure(75)
    for gtr in min_gsigs:
        dob = gtr.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='b')
    for gtr in min_gsigs2:
        dob = gtr.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='y')
    for sig in min_gsigs:
        [new_pos, nlls_var] = gm.gauss_newton(sig, sensors, sig.state_end.mean , cfg.gn_steps, rd_wt)#lm_refine, gauss_newton
        sig.state_end.mean = new_pos
    gr1_centers = []
    plt.figure(75)
    for gtr in min_gsigs:
        dob = gtr.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='r')
        gr1_centers.append(ob.PointTarget(dob[0], dob[1], dob[2], dob[3]))
    pr.plot_scene(plt, scene, sensors, 75, 'Graph pruning detects Relax:{}[blue], MCF:{}[Yellow] targets'.format(len(min_gsigs),len(min_gsigs2)))
    break
    #%%
    t=time.time()
    G1,_ = grpr.make_graph(garda_sel, sensors, True)
    
    [graph_sigs, Ngsig]=grpr.enum_graph_sigs(G1, sensors)
    llr_gr_all=[]
    plt.figure(79)
    for sif in graph_sigs:
        l_cost, g_cost = mle.est_pathllr(sif, sensors, len(sensors)-1, cfgp['rd_wt'], False)
        dob = sif.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='r')
        llr_gr_all.append(l_cost)
        sif.llr = l_cost
    sig_final = sorted(graph_sigs, key=lambda x: x.llr )
    pr.plot_scene(plt, scene, sensors, 79, 'All {} Phantoms'.format(len(graph_sigs)))
    rtime_algo["Brute_all"]= time.time()-t

    grb_centers = []
    plt.figure(78)
    for i in range(Nob):
        gtr = sig_final[i]
        dob = gtr.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='r')
        grb_centers.append(ob.PointTarget(dob[0], dob[1], dob[2], dob[3]))
    pr.plot_scene(plt, scene, sensors, 78, 'Brute Force detects {} targets'.format(len(min_gsigs)))
    
    pr.plot_graph(G1, graph_sigs, sensors, rd_wt, 76, plt, garda_sel, 0)
    pr.plot_graph(G1, min_gsigs, sensors, rd_wt, 77, plt, garda_sel, 0)
#%% Compute error measures
    ospa_error[f,:],_ = np.array(prfe.compute_ospa(scene, gr1_centers, sensors, gardat))
    #%% Average or update scene
    if not static_snapshot: scene = targets_list # Update scene for next timestep

if True:
    #%% [markdown]
    # ## Plot errors in Range, Doppler CRB
    #%% Plot error in range, doppler for this instance
    #plt.figure(38)
    #for pn in range(4):
    #    plt.subplot(2,2,pn+1)
    #    plt.stem(dList[:,pn], cd[:,pn], bottom=0.75);plt.xlabel('Doppler');plt.ylabel('Error')
    
    
    #%% Final Plotting
    # plt.switch_backend('Qt4Agg')  
    plt.figure(11)
    plt.bar(range(3), runtime[:,0], tick_label=alg_name)
    #plt.imshow(np.abs(fft1))
    plt.draw()
    #pickle.dump(plt.figure(1), open("plot.pickle", "wb"))
    #plt.savefig('test.eps',Transparent=True)
    plt.figure(100)
    assgn = zip(*rtime_algo)
    plt.bar(range(len(rtime_algo)), list(rtime_algo.values()), tick_label=list(rtime_algo.keys()))
    # Analyze track quality
    plt.figure(12)
    plt.plot(St_er)
    plt.xlabel('Nf'),plt.ylabel('RMS Error'),plt.title('Error Nearest Phantom(Solid), Auto KF(Dashed)')
    #plt.figure(5)
    plt.plot(Auto_er, linestyle='--'),plt.legend(['x','y','v_x','x','y','v_x'])
    # Ananlyze
    if 1:
        capt13 = ['Overall','Localization','Cardinality']
        plt.figure(13)
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.plot(Noba, ospa_error[:,i], 'bs')
            plt.xlabel('Nf'),plt.ylabel('RMS Error'),plt.title(capt13[i])
    #    for i in range(3,5):
    ##        print(gr1_centers[i-3].x)
    #        print(ospa_error[:,i])
    #%%
    def plotg(res, sigmar):
        count, bins, ignored = plt.hist(res, 20, density=True)
        plt.plot(bins, 1/(sigmar * np.sqrt(2 * np.pi)) *
            np.exp( - (bins)**2 / (2 * sigmar**2) ),
            linewidth=2, color='r')
        plt.show()
if True:        
    capt4 = ['Range Error','Doppler Error']
    plt.figure(14)
    for i in range(2):
        plt.subplot(1,2,i+1)
        for j in range(Nsens):
            plt.errorbar(range(1,Noba[0]+1), rd_err1[j,:,i]/present[j,:], np.sqrt((rd_err2[j,:,i]/present[j,:]-(rd_err1[j,:,i]/present[j,:])**2)))
        plt.xlabel('Nf'),plt.ylabel('RMS Error'),plt.title(capt4[i]),plt.legend(range(1,Noba[0]+1)),plt.grid(True)
    plt.figure(15)
    plt.subplot(1,2,1)
    for j in range(Nsens):
        plt.plot(range(1,Noba[0]+1), present[j,:]/Nf, 'r'),plt.title('Probability of detection')
        plt.xlabel('Target no.'),plt.ylabel('P_D')
    plt.subplot(1,2,2)
    plt.plot(range(1,Nsens+1), Nmiss1/Nf, 'r')
    plt.plot(range(1,Nsens+1), Nfa1/Nf, 'b'),plt.title('Probability of Miss, False Alarm')
    plt.figure(16)
    for i in range(2):
        plt.subplot(1,2,i+1)
        for j in range(Nsens):
            plt.plot(range(1,Noba[0]+1), np.sqrt((rd_err2[j,:,i]/present[j,:]-(rd_err1[j,:,i]/present[j,:])**2)))
        plt.gca().set_prop_cycle(None)# Reset coloring
        plt.subplot(1,2,i+1)
        for j in range(Nsens):
            plt.plot(range(1,Noba[0]+1), np.sqrt(crb1[j,:,i]/present[j,:]), '--')
        plt.xlabel('Sensor'),plt.ylabel('RMS Error'),plt.title(capt4[i]),plt.grid(True),plt.yscale('log')
    plt.figure(17)
    rd_error = np.array(prfe.compute_rd_error(garda_sel, gardat, plt))
    #%%
    plt.figure(18)
    for i in range(Nsens):
        for j in range(Noba[0]):
            plt.subplot(Nsens,Noba[0],i*Noba[0]+j+1)
            tbh = rd_errall[:,i,j,0]
            vidx = np.nonzero(tbh)
            if vidx[0].size>0: 
                plotg(tbh[vidx], np.sqrt(crb1[i,j,0]))
    plt.figure(19)
    plt.plot(glen)
    #%%
    #import pprofile
    #prof = pprofile.Profile()
    #with prof():
    #    garda3 = pr.nomp(np.copy(beat), sensors, [], [4,4], [2,4], 1e-2)
    #prof.callgrind('cachegrind.out.nomp')

    #%%
    importlib.reload(pr)
    t= time.time()
    garda3 = ea.nomp(np.copy(beat), sensors, [], [3,3], [2,4], 1e-2)
    print(time.time()-t)
    #%%
if True: # Huber regression for max flow
    gr_ca=[]
    mode = 'huber'
    for sig in min_gsigs2:
        [era,gr_c]=sig.get_rd_fit_error(sensors, mode, False)
        print(gr_c.x, gr_c.y) #print(era)
        gr_ca.append(gr_c)
    plt.figure(202)
    plt.subplot(1,2,1)
    bp.draw_line_reduction(garda_sel, sensors, scene, gr_ca, [], plt, mode)
    plt.subplot(1,2,2)
    bp.draw_line_reduction2(garda_sel, sensors, scene, gr_ca, [], plt, mode)
    plt.figure(203)
    for gr in gr_ca:
        dob = gr.state
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='r')
    #sif.llr = l_cost
    pr.plot_scene(plt, scene, sensors, 203, 'Detected {} phantoms using {}'.format(len(gr_ca), mode))