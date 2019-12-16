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
sensors = [ob.Sensor(x,0) for x in np.linspace(-2,2,12)]

tf_list = np.array([sensor.mcs.tf for sensor in sensors])  # All sensors frame times equal
tfa_list = np.array([sensor.mcs.get_tfa() for sensor in sensors])  # Adjust so that samples vary to keep frame time const.
Nf = 1 #cfg.Nf
Noba = [15] #cfg.Noba
static_snapshot = 1

## Estimation Parameters
rd_wt = cfg.rd_wt # Range doppler relative weighting for likelihood, NLLS (Selection purposes)

alg_name = ['DFT2', 'Interp DFT2','NOMP']
colr=['r','b','g']
runtime = np.zeros([3,Nf])
rtime_algo = dict()
# snra = np.linspace(-20,10,Nf)
snra = np.ones(Nf)*-10
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

np.random.seed(3)

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
    G1, rtime_make = grpr.make_graph(garda_sel, sensors, rob)
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
    
    if False: # Max flow Association (Not good)
        Gnx, pos,ed_lbl = pr.plot_graph2(G1, graph_sigs, sensors, rd_wt, 78, plt, garda_sel) # All edges
        import networkx as nx
        plt.figure(79)
        nx.draw_networkx(Gnx, pos)
        edge_labels = nx.get_edge_attributes(Gnx,'flow')
        for e in edge_labels:
            edge_labels[e]= int(edge_labels[e])
        nx.draw_networkx_edge_labels(Gnx, pos, label_pos=0.2, edge_labels = edge_labels)  
        plt.figure(80)
        nx.draw_networkx(Gnx, pos)
        edge_labels = nx.get_edge_attributes(Gnx,'capacity')
        for e in edge_labels:
            edge_labels[e]= int(edge_labels[e])
        nx.draw_networkx_edge_labels(Gnx, pos, label_pos=0.7, edge_labels = edge_labels)  
        so_no = sum([len(g) for g in G1])
        t = time.time()
        tracks = pr.max_flow_assoc(Gnx.copy(), so_no, so_no+1)
        min_gsigs = grpr.add_sosi_to_G(G1, Gnx, tracks, sensors)
        print('Max-Flow Association took {}s'.format(time.time()-t))
    #%% MCF Association
    cfgp['rob'] = 2
    cfgp['mode']='mcf_all' # SPEKF, Relax
    from GAutils import mcft as mcft
    t=time.time()
#    min_gsigs3, glen3, rtime_assoc3 = grpr.get_minpaths(cp.deepcopy(G1), sensors, cfgp['mode'], cfgp) # mcft.get_mcfsigs(garda_sel, sensors)
    min_gsigs3, glen3, rtime_assoc3 = mcft.get_mcfsigs(garda_sel, sensors, cfgp)
#    min_gsigs3, glen3, rtime_assoc3 = mcft.get_mcfsigs_all(garda_sel, sensors, cfgp)
    print('{} Association took {}, {}s'.format(cfgp['mode'], rtime_assoc3, time.time()-t))
    pr.plot_graph(G1, min_gsigs3, sensors, rd_wt, 79, plt, garda_sel) # From Relax
    #%%
    
    for sig in min_gsigs1:
        [new_pos, nlls_var] = gm.gauss_newton(sig, sensors, sig.state_end.mean , 5, rd_wt)
        sig.state_end.mean = new_pos
    rtime_algo["Graph"]= time.time()-t
    
    gr_ca = []
    gr_ca2 = []        
    plt.figure(76)
    for gtr in min_gsigs1:
        dob = gtr.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='r')
        gr_ca.append(ob.PointTarget(dob[0], dob[1], dob[2], dob[3]))
#        print(dob, gtr.r)
    pr.plot_scene(plt, scene, sensors, 76, 'GA-DFS detects {} targets'.format(len(min_gsigs1)))
    for sig in min_gsigs3:
        [new_pos, nlls_var] = gm.gauss_newton(sig, sensors, sig.state_end.mean , 5, rd_wt)
        sig.state_end.mean = new_pos
    plt.figure(75)
    for gtr in min_gsigs3:
        dob = gtr.state_end.mean
        plt.quiver(dob[0], dob[1], dob[2], dob[3],color='r')
        gr_ca2.append(ob.PointTarget(dob[0], dob[1], dob[2], dob[3]))
        print(dob, gtr.r)
    pr.plot_scene(plt, scene, sensors, 75, 'Min cost Flow detects {} targets'.format(len(min_gsigs3)))
    #%%
    plt.figure(203)
    plt.subplot(1,2,1)
    bp.draw_line_reduction(garda_sel, sensors, scene, gr_ca2, [], plt, 'Ga-DFS')
    plt.subplot(1,2,2)
    bp.draw_line_reduction2(garda_sel, sensors, scene, gr_ca2, [], plt, 'MCF')
    break # Stop here (Older code ahead)
    
    #########################
    
    #%% Investigating linearity of signatures
    [signatures_raw, Nsig] = am.enumerate_raw_signatures(garda_sel, np.copy(ordered_links), np.copy(forward_links), sensors) # NO thresholding
    print ('Signatures (w/o thres):',len(signatures_raw))

    sig_stda=[]
    sig_stdp=[]
    sig_meana=[]
    ob_raw = []
    sig_raw = []
    ob_rawp = []
    sig_rawp = []
    th_ind = []
    for i, sgi in enumerate(graph_sigs): # Plot all signatures & their llr
        sig_mean, sig_std = bp.compute_linkage_error(sgi, sensors)
        if sig_std < 5:
            th_ind.append(i)
            sig_rawp.append(sgi)
            sig_stdp.append(sig_std)
            ob_rawp.append(ob.PointTarget(sig_mean[0], sig_mean[1], sig_mean[2], sig_mean[3]))
        sig_stda.append(sig_std)
        sig_meana.append(sig_mean)
        sig_raw.append(sgi)
        ob_raw.append(ob.PointTarget(sig_mean[0], sig_mean[1], sig_mean[2], sig_mean[3]))
#        for pos in posM:
#            plt.quiver(pos[0], pos[1], pos[2], pos[3], color ='g')
    plt.figure(25)
    plt.subplot(1,2,1)
    for (sig_mean, sig_std) in zip(sig_meana, sig_stda):
        plt.quiver(sig_mean[0], sig_mean[1], sig_mean[2], sig_mean[3], color ='b', alpha = 1-0.9*((sig_std-min(sig_stda))/(max(sig_stda)-min(sig_stda))), headwidth = 4,headlength =4)
    pr.plot_scene(plt, scene, sensors, 25, 'Linkage errors All {}'.format(len(sig_meana)))

    plt.subplot(1,2,1)
    for ind in th_ind:
        sig_mean= sig_meana[ind]
        sig_std = sig_stda[ind]
        plt.quiver(sig_mean[0], sig_mean[1], sig_mean[2], sig_mean[3], color ='r', alpha = 1-0.9*((sig_std-min(sig_stda))/(max(sig_stda)-min(sig_stda))), headwidth = 4,headlength =4)
    pr.plot_scene(plt, scene, sensors, 25, 'Linkage errors All {}'.format(len(sig_meana)))
    
    # Draw point for all obs
    plt.figure(26)
    plt.subplot(1,3,1)
    bp.draw_line_reduction(garda_sel, sensors, scene, ob_rawp, sig_stdp, plt)
    
    #%%
    t=time.time()
    [signatures, Nsig, ordered_new, pht_all2] = am.enumerate_pruned_signatures(garda_sel, np.copy(ordered_links), np.copy(forward_links), sensors, rd_wt) # With thresholding
    sig_stda=[]
    sig_meana=[]
    for sgi in signatures: # Plot all signatures & their llr
        sig_mean, sig_std = bp.compute_linkage_error(sgi, sensors)
        sig_stda.append(sig_std)
        sig_meana.append(sig_mean)
#        for pos in posM:
#            plt.quiver(pos[0], pos[1], pos[2], pos[3], color ='g')
    plt.figure(25)
    plt.subplot(1,2,2)
    for (sig_mean, sig_std) in zip(sig_meana, sig_stda):
        plt.quiver(sig_mean[0], sig_mean[1], sig_mean[2], sig_mean[3], color ='b', alpha = 1-0.9*((sig_std-min(sig_stda))/(max(sig_stda)-min(sig_stda))), headwidth = 4,headlength =4)
        plt.text(sig_mean[0], sig_mean[1], int(sig_std), FontSize= 8)
    pr.plot_scene(plt, scene, sensors, 25, 'Linkage errors Priuned')
    
    print ('Signatures (w thres):',len(signatures))
    plt.figure(34)
    for pht in pht_all: plt.quiver(pht.x, pht.y, pht.vx, pht.vy, color='g', headwidth = 4) # Plot raw phantoms
    for pht in pht_all2: plt.quiver(pht.x, pht.y, pht.vx, pht.vy, color='b', headwidth = 3) # Plot pruned phatoms
    pr.plot_scene(plt, scene, sensors, 34, '{} Phantoms using consecutive pairs, {} after pruning'.format(len(pht_all), len(pht_all2)))
    pr.plot_orderedlinks(ordered_new, garda_sel, sensors, rd_wt, 22, plt) # Plot pruned graph
#    orli = prfe.get_true_ordered_list(garda_sel,gardat)
#    pr.plot_orderedlinks(orli, garda_sel, sensors, rd_wt, 1, plt) # Plot true graph
    
    sig_all = pr.compute_yparams(sensors, signatures)
    ph_llr = [mle.est_llr(pht, sensors, garda_sel, rd_wt) for pht in pht_all]
    sig_llr = [mle.est_llr(sig_all[s], sensors, garda_sel, rd_wt) for s, sg in enumerate(signatures)]
    
    asc_targets[f] = Nsig
   
    #%% PLot likelihood maps
    if 0:
        [xgrid, ygrid, llr_map] = mle.create_llrmap([-9,9,180], [1,12,110], [-5,5,2], [-5,5,2], sensors, garda_sel) # Position
        cmap = plt.get_cmap('PiYG')
        plt.figure(2)
        im1 = plt.pcolormesh(xgrid, ygrid, llr_map, cmap=cmap)
        plt.colorbar(im1)
        pr.plot_scene(plt, scene, sensors, 2, 'Likelihood Map (Brute Force, Only using r)')
        ## compute llr threshold
        plt.figure(3)
        for spltno, target in enumerate(scene):
            plt.subplot(np.floor(np.sqrt(Nob)),np.ceil(np.sqrt(Nob)),spltno+1)
            [vxgrid, vygrid, llr_map_dop] = mle.create_llrmap_doppler([-9,9,2], [1,12,2], [-8,8,100], [-8,8,100], sensors, garda_sel, target, rd_wt) # Velocity
            im2 = plt.pcolormesh(vxgrid, vygrid, llr_map_dop, cmap=cmap)
            es_id = np.unravel_index(np.argmax(llr_map_dop, axis=None), llr_map_dop.shape)
            plt.scatter(target.vx, target.vy, marker='*', color='k')
            plt.scatter(vxgrid[es_id], vygrid[es_id], marker='d', color='b')
            plt.colorbar(im2)
            plt.title('Target {}'.format(spltno+1))
    ## Brute ML over all 4 dimensions
#    [xgrid_rd, ygrid_rd, vxgrid_rd, vygrid_rd, llr_map_rd] = mle.create_llrmap_rd([-9,9,32], [1,12,16],[-6,6,16], [-6,6,16], sensors, garda_sel)
#    es_id = np.unravel_index(np.argmax(llr_map_rd, axis=None), llr_map_rd.shape)
#    plt.figure(3)
#    plt.quiver(xgrid_rd[es_id[0],es_id[1]],ygrid_rd[es_id[0],es_id[1]],vxgrid_rd[es_id[2],es_id[3]],vygrid_rd[es_id[2],es_id[3]],color='r')
#    plt.figure(12)
#    im12 = plt.pcolormesh(xgrid_rd, ygrid_rd, llr_map_rd[:,:,es_id[2],es_id[3]], cmap=cmap)
#    plt.colorbar(im12)
#    plt.figure(13)
#    im13 = plt.pcolormesh(vxgrid_rd, vygrid_rd, llr_map_rd[es_id[0],es_id[1],:,:], cmap=cmap)
#    plt.colorbar(im13)
#    print('Brute max, x,y,vx,vy= {},{},{},{}'.format(xgrid_rd[es_id[0],es_id[1]],ygrid_rd[es_id[0],es_id[1]],vxgrid_rd[es_id[2],es_id[3]],vygrid_rd[es_id[2],es_id[3]]))
#    break

#%% [markdown]
# ###    #%% Iteratively prune

    #%%
    if False:
        print('Iteratively pruning Phantoms:')
        #    Pht_centroids = cm.cluster_pht1(pht_all, Nsel, ph_llr)# Using all at once
        time.time()
        Pht_centers = itpr.iterative_prune_pht(garda_sel, sensors, Nob, rd_wt, plt)# Iterative phantom based
        rtime_algo["It_Phantoms"]= time.time()-t
        plt.figure(4)
        for pht_idx, pht in enumerate(pht_all):# Plot all phantoms & their likelihood
            (xi, vxi)= (pht.x, pht.vx)
            (yi, vyi) = (pht.y, pht.vy)
            plt.quiver(xi, yi, vxi, vyi, color ='g', alpha = 0.1+0.9*((ph_llr[pht_idx]-min(ph_llr))/(max(ph_llr)-min(ph_llr))), headwidth = 4,headlength =4)
        #    for ce in Pht_centroids: plt.quiver(ce.x,ce.y,ce.vx,ce.vy, color='m', headwidth = 3.5) # Plot centroids
        for ce in Pht_centers: plt.quiver(ce.x,ce.y,ce.vx,ce.vy, color='r', headwidth = 3) # Plot iteratively selected loc. estimates
        pr.plot_scene(plt, scene, sensors, 4, 'Likelihood at {} raw Phantoms using all (r,d) pairs'.format(N_pht))
    
    #
        print('Iteratively pruning Signatures:')
        #    sig_centroids = cm.cluster_sig1(signatures, Nsel, sensors, sig_llr) # Using all tracks at once
        time.time()
        sig_centers = itpr.iterative_prune_sig(garda_sel, sensors, Nob, rd_wt) # Iterative track based
        rtime_algo["It_Signature"]= time.time()-t
        plt.figure(5)
        for sgi, obc in enumerate(sig_all): # Plot all signatures & their llr
            plt.quiver(obc.x, obc.y, obc.vx, obc.vy, color ='b', alpha = 0.1+0.9*((sig_llr[sgi]-min(sig_llr))/(max(sig_llr)-min(sig_llr))), headwidth = 4,headlength =4)
        #    for ce in sig_centroids: plt.quiver(ce.x,ce.y,ce.vx,ce.vy, color='m', headwidth = 3.5) # Plot centroids
        for ce in sig_centers: plt.quiver(ce.x,ce.y,ce.vx,ce.vy, color='r', headwidth = 3) # Plot iteratively selected estimates
        pr.plot_scene(plt, scene, sensors, 5, 'Likelihood at KF Track estimates ({} tracks, using (r,d))'.format(len(signatures)))

#%%


#%% [markdown]
# ## Begin Estimation

#%%
    #%% Estimate using NLLS on all tracks
    L_thres = mle.compute_llr_thres(sensors, rd_wt)
    thres2 = mle.compute_llr_thres2(sensors, garda_sel, rd_wt)
    th_ind = (np.squeeze(np.nonzero(sig_llr>thres2)))
    nlls_est = [[] for si in th_ind]
    nlls_var = np.zeros(len(th_ind))
    nlls_all =[]
    for i, si in enumerate(th_ind):
        sg = signatures[si]
        [nlls_est[i], nlls_var[i]] = gm.gauss_newton(sg, sensors, sig_all[si].state , 5, rd_wt)#[sg.state_head.mean[0],y_est_sig[si], sg.state_head.mean[1],vy_est_sig[si]]
        xi,yi,vxi,vyi = nlls_est[i].tolist()
        nlls_all.append(ob.PointTarget(xi,yi,vxi,vyi))
    nlls_sorted_idx = np.argsort(nlls_var)
    #    for si in nlls_sorted_idx[0:min(Nsig, Nsel)]:
    #        plt.quiver(nlls_est[si][0],nlls_est[si][1],nlls_est[si][2],nlls_est[si][3],color='r', alpha=1-0.9*((nlls_var[si]-min(nlls_var))/(max(nlls_var)-min(nlls_var))), headwidth = 3)
    plt.figure(7)
    for si in nlls_sorted_idx: # Plot all signatures & their llr
        plt.quiver(nlls_est[si][0],nlls_est[si][1],nlls_est[si][2],nlls_est[si][3],color='r', alpha=1-0.9*((nlls_var[si]-min(nlls_var))/(max(nlls_var)-min(nlls_var))), headwidth = 4)
    pr.plot_scene(plt, scene, sensors, 7, 'All {} tracks refined using NLLS refinements'.format(len(signatures)))
    #    plt.figure(8)
    #    for i in range(3):
    #        plt.subplot(3,1,1+i)
    #        nlls_centroids = cm.cluster_pht1(nlls_all, Nsel, np.ones(len(nlls_all)), i)
    #        for ce in nlls_centroids: plt.quiver(ce.x,ce.y,ce.vx,ce.vy, color=colr[i], headwidth = 3.5) # Plot centroids
    #        pr.plot_scene(plt, scene, sensors, 8, '{} Clustering with {} refined est'.format(cluster_algo[i],len(nlls_all)))
    #    
    #%% Refining Raw signatures pruned by track variation
    raw_refined = []
    raw_llr = []
    for i, sg in enumerate(sig_rawp):
        [nlls_raw, rllr] = gm.gauss_newton(sg, sensors, ob_rawp[i].state , 5, rd_wt)#[sg.state_head.mean[0],y_est_sig[si], sg.state_head.mean[1],vy_est_sig[si]]
        xi,yi,vxi,vyi = nlls_raw.tolist()
        raw_refined.append(ob.PointTarget(xi,yi,vxi,vyi))
        raw_llr.append(rllr)
    plt.figure(8)
    for si in range(len(raw_llr)): # Plot all signatures & their llr
        plt.quiver(raw_refined[si].x,raw_refined[si].y,raw_refined[si].vx,raw_refined[si].vy,color='r', alpha=1-0.9*((raw_llr[si]-min(raw_llr))/(max(raw_llr)-min(raw_llr))), headwidth = 4)
    pr.plot_scene(plt, scene, sensors, 8, ' {} tracks refined using NLLS refinements'.format(len(raw_llr)))
    #%%
    plt.figure(28)
    plt.subplot(1,3,1)
    bp.draw_line_reduction(garda_sel, sensors, scene, raw_refined, raw_llr, plt, 'Track purity')
    plt.subplot(1,3,2)
    bp.draw_line_reduction(garda_sel, sensors, scene, Pht_centers, [], plt, 'Phantom')
    plt.subplot(1,3,3)
    bp.draw_line_reduction(garda_sel, sensors, scene, sig_centers, [], plt, 'Signature')
    plt.figure(29)
    #    bp.draw_link_reduction()
    erra_lin = bp.compute_linearity_error(sig_rawp, sensors, 0.1)
    for i, sg in enumerate(sig_rawp):
        x_val=[sensors[si].x for si in sg.sindx]
        y_val = sg.r * sg.d
    #        plt.subplot(1,2,1)
        p2 = plt.plot(x_val, y_val, 'r-', alpha = 1-0.9*(raw_llr[i]-min(raw_llr))/(max(raw_llr)-min(raw_llr)))
        
        for i, gard in enumerate(garda_sel):
            yp = gard.r * gard.d
            xp = (2*i-3) * np.ones(len(yp))
            p1 = plt.plot(xp, yp, 'bo')
    #        plt.subplot(1,2,2)
    #        p2 = plt.plot(x_val, y_val, 'r-', alpha = 1-0.9*(erra_lin[i]-min(erra_lin))/(max(erra_lin)-min(erra_lin)))
    x_val = np.array([sensor.x for sensor in sensors])#2*np.arange(4)-3
    
#    for obj in scene:
#        r_true = np.sqrt(obj.x**2 + obj.y**2)
#        d_true = (obj.x * obj.vx + obj.y * obj.vy)/r_true
#        y_val = -x_val * obj.vx + r_true*d_true
#        p3 = plt.plot(x_val, y_val, 'k.:')
#    plt.grid(True)
#    time.time()
#    tp_centers = bp.iterative_tp_pruning(sig_rawp, sensors, 10, rd_wt)
#    rtime_algo["It_GeoPrune"]= time.time()-t
#    plt.figure(9)
#    for ce in tp_centers: plt.quiver(ce.x,ce.y,ce.vx,ce.vy, color='r', headwidth = 3.5) # Plot centroids
#    pr.plot_scene(plt, scene, sensors, 9, 'Track purity result'.format(len(tp_centers)))
    
    #%%
    importlib.reload(bp)
    #    bp.draw_link_reduction()
    erra_lin2, Qph = bp.compute_closeness(sig_raw, sensors, 1, 20)
    err_llr = [mle.est_llr(temp_obj, sensors, garda_sel, [1,1]) for temp_obj in Qph]
    
    
    #%%
    plt.figure(49)
    for si in range(len(sig_raw)): # Plot all signatures & their llr
        if erra_lin2[si]<50:
            plt.quiver(Qph[si].x, Qph[si].y, Qph[si].vx, Qph[si].vy, color='b', 
                   alpha=1-0.9*((erra_lin2[si]-min(erra_lin2))/(max(erra_lin2)-min(erra_lin2))), headwidth = 4)
    pr.plot_scene(plt, scene, sensors, 49, 'Tracks extracted with minimum fitting error'.format(len(signatures)))
    
    
    #%%
    plt.figure(50)
    for si in range(len(sig_raw)): # Plot all signatures & their llr
        if 1:
            plt.quiver(Qph[si].x, Qph[si].y, Qph[si].vx, Qph[si].vy, color='b', 
                   alpha=0.1+0.9*((err_llr[si]-min(err_llr))/(max(err_llr)-min(err_llr))), headwidth = 4)
    pr.plot_scene(plt, scene, sensors, 50, 'Tracks extracted with minimum fitting error'.format(len(signatures)))
    
    #%% Check runtime of elementary solutions
    import cvxpy as cp
    points =[]
    points2 =[]
    err_val1 = []
    err_val2 = []
    time.time()
    for i, sg in enumerate(sig_raw):
        Me = sg.r * sg.d
        Me2 = sg.r * sg.r
        L = np.array([sensors[si].x for si in sg.sindx])
        Ns = len(sg.r)
        Z_mat = np.eye(Ns)
        Z = Z_mat[0:-1,:]-Z_mat[1:,:]
    #     Z = np.vstack((Z,[1,0,-1,0],[0,1,0,-1],[1,0,0,-1])) # Add all other pairs
        # Form and solve a standard regression problem.
        beta_x = cp.Variable(1)
        beta_vx = cp.Variable(1)
    #     fit = norm(beta - beta_true)/norm(beta_true)
        cost = (cp.norm(2*beta_x*(L@Z.T) - (L*L)@Z.T + Me2@Z.T )
               +cp.norm(beta_vx*(L@Z.T) + Me@Z.T ))
        constraints = [beta_x - sg.r <= 0] 
    #                   -sg.r - beta_x <= 0]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        x_hat = beta_x.value
        v_hat = beta_vx.value
        
        xsa = x_hat - L
        y_est = np.sqrt(np.mean(Me2 - xsa **2))
        vy_est = np.mean(Me - v_hat*xsa) / y_est # Estimated using other estimates
        points.append(ob.PointTarget(x_hat, y_est, v_hat, vy_est))
        err_val1.append(cost.value)
    
        # Form and solve the Huber regression problem.
        cost = (cp.atoms.sum(cp.huber(2*beta_x*(L@Z.T) - (L*L)@Z.T + Me2@Z.T, 5))
               + cp.atoms.sum(cp.huber(beta_vx*(L@Z.T) + Me@Z.T, 5)))
        cp.Problem(cp.Minimize(cost)).solve()
        x_hat = beta_x.value
        v_hat = beta_vx.value
        
        xsa = x_hat - L
        y_est = np.sqrt(np.mean(Me2 - xsa **2))
        vy_est = np.mean(Me - v_hat*xsa) / y_est # Estimated using other estimates
        
        err_val2.append(cost.value)
    #     huber_data[idx] = fit.value
        points2.append(ob.PointTarget(x_hat, y_est, v_hat, vy_est))
    rtime_algo['CVX']= (time.time()-t)/2
    print(rtime_algo['CVX'])
    #%% Compare errors of all approaches
    # Create traces
    from plotly.offline import plot
    trace0 = go.Scatter(
        x = np.arange(len(erra_lin2)),
        y = erra_lin2,
        mode = 'lines',
        name = 'Dual Max'
    )
    trace1 = go.Scatter(
        x = np.arange(len(erra_lin2)),
        y = (np.amax(err_llr)-np.array(err_llr))/100,
        mode = 'lines+markers',
        name = 'Geometric'
    )
    trace2 = go.Scatter(
        x = np.arange(len(erra_lin2)),
        y = err_val2,
        mode = 'markers',
        name = 'Huber'
    )
    data = [trace0, trace1, trace2]
    
    plot(data, filename='line-mode')
    
    
    #%% PLotting CVX results
    plt.figure(58)
    ind58a=[]
    plt.subplot(1,2,1)
    for si in range(len(sig_raw)): # Plot all signatures & their llr
        if err_val1[si]<15: #5*min(err_val1):
            ind58a.append(si)
            plt.quiver(points[si].x, points[si].y, points[si].vx, points[si].vy, color='r', 
                   alpha=1-0.9*((err_val1[si]-min(err_val1))/(max(err_val1)-min(err_val1))), headwidth = 4)
    pr.plot_scene(plt, scene, sensors, 58, 'Tracks extracted with CVX (LS) minimum fitting error'.format(len(signatures)))
    plt.subplot(1,2,2)
    for si in range(len(sig_raw)): # Plot all signatures & their llr
        if err_val2[si]<50: #*min(err_val2):
            plt.quiver(points2[si].x, points2[si].y, points2[si].vx, points2[si].vy, color='b', 
                   alpha=1-0.9*((err_val2[si]-min(err_val2))/(max(err_val2)-min(err_val2))), headwidth = 4)
    pr.plot_scene(plt, scene, sensors, 58, 'Tracks extracted with CVX (Huber) minimum fitting error'.format(len(signatures)))

#%% Compute error measures
    ospa_error[f,:] = np.array(prfe.compute_ospa(scene, sig_centers))
    #%% Average or update scene
    if not static_snapshot: scene = targets_list # Update scene for next timestep

#%% [markdown]
# ## Plot errors in Range, Doppler CRB
#%% Plot error in range, doppler for this instance
#plt.figure(38)
#for pn in range(4):
#    plt.subplot(2,2,pn+1)
#    plt.stem(dList[:,pn], cd[:,pn], bottom=0.75);plt.xlabel('Doppler');plt.ylabel('Error')


#%% Final Plotting
if 0:
    # plt.switch_backend('Qt4Agg')  
    plt.figure(11)
    plt.bar(range(3), np.mean(runtime, 1), tick_label=alg_name)
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
    capt13 = ['Overall','Localization','Cardinality']
    plt.figure(13)
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.plot(Noba, ospa_error[:,i], 'bs')
        plt.xlabel('Nf'),plt.ylabel('RMS Error'),plt.title(capt13[i])
    
    #capt4 = ['Range Error','Doppler Error']
    #plt.figure(14)
    #for i in range(2):
    #    plt.subplot(1,2,i+1)
    #    plt.errorbar(Noba, np.mean(rd_error[:,i], axis =1), np.std(rd_error[:,i], axis =1))
    #    plt.xlabel('Nf'),plt.ylabel('RMS Error'),plt.title(capt4[i])
    #plt.show()

# if __name__ == "__main__":
#	main()