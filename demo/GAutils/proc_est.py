#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 12:27:21 2018

@author: anantgupta
"""
from __future__ import division
# Add classes for Extended Targets
import numpy as np
#from numba import jit
from GAutils import objects as obt
import matplotlib as mpl
# import networkx as nx
import GAutils.ml_est as mle

def ProcDyms(target, dt, tfas=0):
    # Return delays at indices and target after frame
    AbsPos = np.zeros(tfas.shape + (2,))  # Add dimension 2 to tuple
    for i in range(tfas.shape[0]):
        AbsPos[i, :, :, 0] = target.x + target.vx * (tfas[i])
        AbsPos[i, :, :, 1] = target.y + target.vy * (tfas[i])
    target.x = target.x + target.vx * dt 
    target.y = target.y + target.vy * dt
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    target.state = F*target.state + np.sqrt(target.proc_var)*np.random.randn(4)
    return target, AbsPos

def check_minsep(scene, sensors, garda, newtarget, sep_th, dop=True):
    if not len(garda):
        update_garda = [get_gard_true(sensor, newtarget) for sensor in sensors]
        return True, update_garda
    update_garda = np.copy(garda)
    for gard, sensor in zip(update_garda, sensors):
        R_res = sensor.mcs.c/(2*sensor.mcs.B)
        D_res = sensor.mcs.c/(2*sensor.mcs.fc * sensor.mcs.Ni * sensor.mcs.Nch * sensor.mcs.Ts)
        if dop: beta = np.linalg.inv(np.array([[R_res**2*3/2, -D_res**2],[-D_res**2, D_res**2*3/2]]))# Obtained [[1,-0.6],[-0.6,0.6]] from fitting
        else: beta = np.linalg.inv(np.array([[R_res**2, -min(D_res,R_res)**2],[-min(D_res, R_res)**2, D_res**2]]))# Obtained [[1,-0.6],[-0.6,0.6]] from fitting
        newgard = get_gard_true(sensor, newtarget)
        sep = np.vstack([(newgard.r - gard.r),(newgard.d - gard.d)])
        # if any( ( abs(newgard.r - gard.r)/R_res # Old ellipse boundary
        #          +int(dop)* abs(newgard.d - gard.d)/D_res )<sep_th ) :
        if any( ( (newgard.r - gard.r)**2 * beta[0,0] + # Tight/Loose Ellipse boundary
                    (newgard.d - gard.d)**2 * beta[1,1] +
                 +(newgard.d - gard.d)*(newgard.r - gard.r)*2*beta[0,1] )<sep_th ):
            return False, garda
        else:
            gard.add_Est(newgard.g, newgard.a, newgard.r, newgard.d)
    return True, update_garda

def init_random_scene(Nt, sensors, sep_th = 0, seed_val = []):
    scene=[]
    garda = []
    if seed_val:# was 53 #np.random.randint(100)
#        print('Seed= ',str(seed_val))
        np.random.seed(seed_val)# was 42
    while len(scene)<Nt:
        newtarget = obt.PointTarget(-8+np.random.rand(1) *16, 2+np.random.rand(1) *10, 20 * np.random.rand(1) -10, 20*np.random.rand(1) -10, 0, 1)
        [valid, garda] = check_minsep(scene, sensors, garda, newtarget, sep_th, False)
        if valid: scene.append(newtarget)
    return scene

def get_amplitude(sensor, target):
    amp = 1  # Should be related to distance between sensor & target
    return amp

def add_cnoise(x, std):# std in (10 **(-snr/20)*signal_power)
    shpe = x.shape
    y = x + std/np.sqrt(2)*(np.random.randn(shpe[0],shpe[1])+1j*np.random.randn(shpe[0],shpe[1]))/np.sqrt(x.size)
    return y
    
def plot_scene(plt, targets, sensors, number, fig_title):
    plt.figure(number)
    for sensor in sensors:
        plt.text(sensor.x, sensor.y, 'Y')
        # plt.quiver(sensor.x, sensor.y, sensor.vx, sensor.vy)
    j=0
    for target in targets:
        plt.quiver(target.x, target.y, target.vx, target.vy, headwidth = 2, linewidths =1)
        plt.text(target.x, target.y, j+1)
        j= j + 1
    plt.axis([-8.5,8.5,0,13])
    plt.grid()#,plt.axis('equal')
    plt.title(fig_title)
    
def get_pos_from_rd(ri, rj, di, dj, i , j, sensors):
# Compute xyvxvy from r,d and sensors locations
    pht_tuple = triangulation(ri,di,sensors[i].x, rj, dj, sensors[j].x)
    
    if pht_tuple is not None: # NOTE: Discard phantoms if y can't be calculated
        (x_est,yr,vx,vyr) = pht_tuple
        pht = obt.PointTarget(x_est,yr,vx,vyr)
        return pht
    else:
        return None

#@jit(nopython=True, cache = True)
def triangulation(ri, di, xi, rj, dj, xj):
    sensor_sep= (xj - xi) # should be euclidean distance if y~=0
    # coordinates along axis of ellipse with foci at sensors
    x_r = (ri**2 - rj**2) / (2 * sensor_sep) #+ sensor_sep[j]/2
    x_est = x_r + (xi + xj)/2  # shift x for linear sensor array
    vx = (ri * di - rj * dj) / sensor_sep 
    y_r2=(ri**2 + rj**2 - (sensor_sep**2)/2 - 2* (abs(x_r)**2))/2 # Square y
    if y_r2>=0: # NOTE: Discard phantoms if y can't be calculated
        yr = np.sqrt( y_r2 ) # y estimate common across all 
        if yr==0:
            vyr = np.inf
        else:
            vyr = ( (ri*di +rj*dj )/2 - vx*x_r) / yr
        pht = (x_est,yr,vx,vyr)
        return pht
    else:
        return None


def plot_orderedlinks(ordered_links, garda, sensors, rd_wt, fignum, plt):
    G = nx.DiGraph()
    Ns = len(garda)
    s = Ns-1
    Lmax = max([len(gard.r) for gard in garda])
    node_id = 0
    node_idp = 0
    pos={}
    edge_alphas=[]
    mle_llr =[]
    for s in range(Ns):
        L =len(garda[s].r)
        for l in range(L): 
            G.add_node(node_id+l )
            pos.update({node_id+l:[s/Ns,l/Lmax]})
        for p, link in enumerate(ordered_links[s]):
             # Add node ids from last sensor
             for ip in link.indx:
                 trg = get_pos_from_rd(garda[s-1].r[ip],garda[s].r[p],garda[s-1].d[ip],garda[s].d[p],s-1,s,sensors)
                 mlellr = mle.est_llr(trg, sensors, garda, rd_wt)
                 mle_llr.append(mlellr)
                 G.add_weighted_edges_from([(node_id + p, node_idp+ip, mlellr)])
#                 plt.quiver(trg.x, trg.y, trg.vx, trg.vy, color ='r', alpha = mle.est_llr(trg, sensors, garda)/20)# Plots the pht
#             G.add_weighted_edges_from([node_id + p,node_idp+ip, 
#                                        mle.est_llr(get_pos_from_rd(garda[s].r[p],garda[s-1].r[ip],garda[s].d[p],garda[s-1].d[ip],s,s-1,sensors), sensors, garda)] for ip in link.indx) # NOTE: Add llr of edges
        node_idp = node_id
        node_id += L
    #%% Plotting
    Amax = np.amax(mle_llr)
    Amin = np.amin(mle_llr)
    
    node_sizes = [10 for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = [Amin + m/M*(Amax-Amin) for m in range(0, M)]

    for (u,v,d) in G.edges(data='weight'):
        edge_alphas.append(d/Amax)

    plt.figure(fignum)
    nx.draw_networkx(G, pos, edge_color=edge_alphas,
                               edge_cmap=plt.cm.YlOrRd, width=1.5)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                               arrowsize=10, edge_color=edge_alphas,
                               edge_cmap=plt.cm.YlOrRd, width=1.5)
    # set alpha value for each edge,  edge_color=edge_colors,
#    i=0
#    for (u,v,d) in G.edges(data='weight'):
#        print(u,v,d,d/Amax)
##        edges[i].set_alpha(d/Amax)
##        edges[i].set_color('blue')
#        edge_alphas.append(d/Amax)
#        i=i+1

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.YlOrRd)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

def plot_graph(Gin, sigs, sensors, rd_wt, fignum, plt, garda, mode=0):
    G = nx.DiGraph()
    Ns = len(sensors)
    s = Ns-1
    Lmax = max([len(Ginc) for Ginc in Gin])
    node_id = [0]
    pos={}
    edge_alphas=[]
    mle_llr =[]
    ra = []
    for s in range(Ns):
        L =len(Gin[s])
        ra.append([Gin[s][i].r for i in range(L)])
        for l in range(L): 
            G.add_node(node_id[s]+l )
            pos.update({node_id[s]+l:[s/Ns,l/Lmax]})
        node_id += [node_id[s]+L]
    for sig in sigs:
        for i in range(sig.N-1):
            sid = sig.sindx[i] # Source sensor id
            did = sig.sindx[i+1] # Dest sensor id
            sr = sig.r[i]
            sd = sig.d[i]
            spid = node_id[sid]+ra[sid].index(sr)# Source node id
            dr = sig.r[i+1]
            dd = sig.d[i+1]
            dpid = node_id[did]+ra[did].index(dr)# Dest node id
             # Add node ids from last sensor
            if mode==0: # Compute arc llr using garda
                trg = get_pos_from_rd(sr,dr,sd,dd,sid,did,sensors)
                mlellr = mle.est_llr(trg, sensors, garda, rd_wt)
            elif mode==1:
                mlellr = -np.log(abs(sig.llr[0]))
            else:
                mlellr = -np.log(abs(sig.gc[i]))
#            if sig.N > 2:
##                print(sig.llr, sig.gc)
#                mlellr =sig.llr[0] /(1+np.exp(-abs(sig.gc[i])))
#            else:
#                mlellr = 0
            mle_llr.append(mlellr)
            G.add_weighted_edges_from([(spid, dpid, mlellr)])
    #%% Plotting
    Amax = np.amax(mle_llr)
    Amin = np.amin(mle_llr)
    
    node_sizes = [10 for i in range(len(G))]
    M = G.number_of_edges()
    rng = Amax-Amin
    edge_colors = [Amin + m/M*(rng) for m in range(0, M)]

    for (u,v,d) in G.edges(data='weight'):
        edge_alphas.append(d/Amax)

    plt.figure(fignum)
    nx.draw_networkx(G, pos,  edge_color=edge_alphas,
                               edge_cmap=plt.cm.copper_r, width=0.2)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                               arrowsize=10, edge_color=edge_alphas,
                               edge_cmap=plt.cm.copper_r, width=0.2+1.8*np.array(edge_alphas))
    # set alpha value for each edge,  edge_color=edge_colors,
#    i=0
#    for (u,v,d) in G.edges(data='weight'):
#        print(u,v,d,d/Amax)
##        edges[i].set_alpha(d/Amax)
##        edges[i].set_color('blue')
#        edge_alphas.append(d/Amax)
#        i=i+1

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.copper_r)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()
    
def get_beat(sensor, target, AbsP, rworld = False):
    # Measures one FMCW frame 
    amp = get_amplitude(sensor, target)
    if rworld:
        tfa = sensor.mcs.get_tfa()
        tna = np.outer(np.ones([tfa.shape[0],1]),tfa[0])
        tda = np.sqrt((AbsP[:, :, 0] - sensor.x) ** 2 + (AbsP[:, :, 1] - sensor.y) ** 2) * 2 / 3e8 # tau
        fbeat = sensor.mcs.ss * tda
        beat = amp * np.exp(1j * 2 * np.pi * (sensor.mcs.fc * tda + tna * fbeat )) / np.sqrt(tfa.size) #- fbeat * tda / 2
        return beat
    else:
        gard_true = get_gard_true(sensor, target, rworld)
        d1 = gard_true.d
        r1 = gard_true.r
#    print('Truth R={}, D={}'.format(r1,d1))
        c=3e8
        rf = c / (2 * sensor.mcs.B)
        df = c / (2 * sensor.mcs.tf * sensor.mcs.fc)
        Ni, Nch = sensor.mcs.Ni, sensor.mcs.Nch
        tfa2 = sensor.mcs.get_tfa(r1 / rf / Ni, d1 / df / Ni / Nch) / sensor.mcs.Ts
        beat2 = amp * np.exp(1j * 2 * np.pi * tfa2) / np.sqrt(tfa2.size)
        return beat2

def get_gard_true(sensor, target, rworld = False):#single target,sensor
    # sensor : Reference Sensor
    # target : observed (single) target 
    xr = target.x - sensor.x
    yr = target.y - sensor.y
    vxr = target.vx - sensor.vx
    vyr = target.vy - sensor.vy
    r0 = np.sqrt( xr ** 2 + yr ** 2)
    d0 = ( xr * vxr + yr * vyr ) / r0
    rc = r0 + int(rworld) * 0.5 * sensor.mcs.tf * d0
    dc = d0 + int(rworld) * 0.5 * sensor.mcs.tf * ( vxr**2 + vyr**2 - d0**2 ) / r0 
    g = get_amplitude(sensor, target )
    gard_true = obt.gardEst()
    gard_true.add_Est(g, 0, rc, dc)
    return gard_true

def compute_yparams(sensors, signatures, same_size=True): # for new method
    sig_all=[]
    for s, signature in enumerate(signatures):
        if 1: # TODO: CAn reject high error signatures here!!
            ob = signature.state_end.mean
            sig_all.append(obt.PointTarget(ob[0],ob[1],ob[2],ob[3]))
    return sig_all

def compute_yparams2(sensors, signatures, same_size=True): # Only for old method
    sig_all=[]
    for s, signature in enumerate(signatures):
        xsa = signature.state_end.mean[0] - [sensors[sid].x for sid in signature.sindx]
        ys_est2 = signature.r **2 - xsa **2 # squared y estimate at sensors
        if any(ys_est2 < 0) & same_size:
            y_est=0
            vy_est = 0
            sig_all.append(obt.PointTarget(signature.state_end.mean[0], y_est, signature.state_end.mean[1], vy_est))
        else:
            y_est = np.sqrt(np.mean(ys_est2))
            vy_est = np.mean(signature.r*signature.d - signature.state_end.mean[1]*xsa) / y_est # Estimated using other estimates
            sig_all.append(obt.PointTarget(signature.state_end.mean[0], y_est, signature.state_end.mean[1], vy_est))
    return sig_all
# def traingulate(r1, r2, d1, d2, lc, lij):
#     xr = (r1**2 - r2**2) / (2 * lij)
#     ys = np.sqrt( (r1**2 + rj**2 - (sensor_sep[j]**2)/2 - 2* (abs(x_r)**2))/2 )

## Elementary MAx Flow 
def plot_graph2(Gin, sigs, sensors, rd_wt, fignum, plt, garda, mode=0):
    G = nx.DiGraph()
    Ns = len(sensors)
    s = Ns-1
    Lmax = max([len(Ginc) for Ginc in Gin])
    node_id = [0]
    pos={}
    ed_lbl={}
    edge_alphas=[]
    mle_llr =[]
    ra = []
    
    for s in range(Ns):
        L =len(Gin[s])
        ra.append([Gin[s][i].r for i in range(L)])
        for l in range(L): 
            G.add_node(node_id[s]+l, ids=[s,l] )
            pos.update({node_id[s]+l:[s/Ns,l/Lmax]})
#            if s==Ns-1: # Sink
#                G.add_edge(so_no, node_id[s]+l, capacity=CMAX)
#            if s==0:
#                G.add_edge(node_id[s]+l, si_no, capacity=CMAX)
        node_id += [node_id[s]+L]

    for sig in sigs:
        for i in range(sig.N-1):
            sid = sig.sindx[i] # Source sensor id
            did = sig.sindx[i+1] # Dest sensor id
            sr = sig.r[i]
            sd = sig.d[i]
            spid = node_id[sid]+ra[sid].index(sr)# Source node id
            dr = sig.r[i+1]
            dd = sig.d[i+1]
            dpid = node_id[did]+ra[did].index(dr)# Dest node id
             # Add node ids from last sensor
            if mode==0: # Compute arc llr using garda
                trg = get_pos_from_rd(sr,dr,sd,dd,sid,did,sensors)
                mlellr = mle.est_llr(trg, sensors, garda, rd_wt)
            elif mode==1:
                mlellr = -np.log(abs(sig.llr[0]))
            else:
                mlellr = -np.log(abs(sig.gc[i]))
#            if sig.N > 2:
##                print(sig.llr, sig.gc)
#                mlellr =sig.llr[0] /(1+np.exp(-abs(sig.gc[i])))
#            else:
#                mlellr = 0
            mle_llr.append(mlellr)
            G.add_edge(spid, dpid, capacity=mlellr)
            ed_lbl[(spid, dpid)]=mlellr
    
    
    
    #%% Add source and sink
    Amax = np.amax(mle_llr)
    so_no= sum([len(g) for g in Gin])
    si_no=so_no+1
    G.add_node(so_no)
    pos.update({so_no:[1,0.37]})
    G.add_node(si_no)
    pos.update({si_no:[-0.25,0.37]})    
    #Add source edges
    for l in range(len(Gin[0])):
        G.add_edge(l, si_no, capacity=Amax+1)
    # Add sink edges
    for l in range(len(Gin[-1])):
        G.add_edge(so_no, so_no-1-l, capacity=Amax+1)
        
    G2=G.copy()
    G = nx.algorithms.flow.edmonds_karp(G, so_no, si_no)
    #%% Plotting
    Amin = np.amin(mle_llr)
    node_sizes = [10 for i in range(len(G))]
    M = G.number_of_edges()
    rng = Amax-Amin
    edge_colors = [Amin + m/M*(rng) for m in range(0, M)]

    for (u,v,d) in G.edges(data='capacity'):
        edge_alphas.append(d/Amax)

    plt.figure(fignum)
#    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='black')
    nx.draw_networkx(G, pos, edge_color=edge_alphas,
                               edge_cmap=plt.cm.copper_r,)
#    nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                               arrowsize=10, edge_color=edge_alphas,
                               edge_cmap=plt.cm.copper_r, width=1.5, label=edge_alphas)
    # set alpha value for each edge,  edge_color=edge_colors,
#    i=0
#    for (u,v,d) in G.edges(data='weight'):
#        print(u,v,d,d/Amax)
##        edges[i].set_alpha(d/Amax)
##        edges[i].set_color('blue')
#        edge_alphas.append(d/Amax)
#        i=i+1

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.copper_r)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()
    return G2,pos,ed_lbl
    
    
def max_flow_assoc(Gnx, so_no, si_no):
    #Perform max flow asocaition, not recommended (MCF is superior)
    tracks = []
    while nx.maximum_flow(Gnx,so_no, si_no)[0]>0:
        Gnx = nx.algorithms.flow.edmonds_karp(Gnx, so_no, si_no)
        new_path = nx.astar_path(Gnx,si_no,so_no, weight='flow')
        tracks.append(new_path[1:-1])
        Gnx.remove_nodes_from(new_path[1:-1])
    return tracks
