# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:54:50 2019
Contains function for operating on Graph
@author: gupta
"""
import numpy as np

# Custom libs
from GAutils import objects as ob
import copy as cp
from GAutils import proc_est as pr
from GAutils import ml_est as mle
# import config as cfg
from scipy.stats import chi2
import heapq, collections

def make_graph(garda, sensors, lskp=False, l2p=0):
    Ns=len(sensors)
    tol = -1e-10 # tolerance for range bands (-ve implies narrower FOV)
    Lp1, Lp2, Lt1, Lt2, Nnodes =0,0, 1,1,0
    G=[]
    for sid, gard in enumerate(garda):
        L=len(gard.g)
        G.append([ob.obs_node(gard.g[oid],gard.a[oid],abs(gard.r[oid]),gard.d[oid],oid, sid) for oid in range(L)])
        Lp1-=L*(L-1)/2 # Num of self edges to subtract later
        Lt1*=L # Total tracks without prunung
        Nnodes +=L
    Lp1+= Nnodes*(Nnodes-1)/2
    for i in range(1,Ns): 
        sobs_c = G[i] # ranges of current sensor
        j=i-1
        sobs_b = G[j]
        l1 = np.sqrt((sensors[i].x - sensors[j].x)**2+(sensors[i].y - sensors[j].y)**2) # sensor separation
        d = sensors[i].fov * l1 + tol # max range delta
        for sob_c in sobs_c:
            for sob_b in sobs_b:
                if abs(sob_b.r-sob_c.r)<d and abs(sob_b.r+sob_c.r)>d :
                    sob_c.insert_blink(sob_b)
                    sob_b.insert_flink(sob_c)
                    Lp2+=1
                    Lt2+=1+len(sob_b.lkb)
            if 0:
            #Add connection jumping across 1 sensor (more sensors?) 
            # Implement using hashmap which remembers track positions
                if lskp and i-2>=0: 
                    k=i-2
                    l2 = np.sqrt((sensors[i].x - sensors[k].x)**2+(sensors[i].y - sensors[k].y)**2)
                    djk = np.sqrt((sensors[j].x - sensors[k].x)**2+(sensors[j].y - sensors[k].y)**2)
                    dk = sensors[i].fov * l2 + tol # max range delta
                    sobs_k = G[k]
                    for sob_k in sobs_k:
                        if (abs(sob_k.r-sob_c.r)<dk and abs(sob_k.r+sob_c.r)>dk and not 
                            any([similar_paths(sob_c, sob_b, sob_k, sensors) for sob_b in sobs_b])):
                            sob_c.insert_blink(sob_k)
                            sob_k.insert_flink(sob_c)
                            Lp2+=1
                            Lt2+=1+len(sob_k.lkb)
            if 1:
                # Recursive implementation for P-skip connections
                k=i-2 # current index of backtracking
                while i-k<=lskp+1 and k>=0:
                    l2 = np.sqrt((sensors[i].x - sensors[k].x)**2+(sensors[i].y - sensors[k].y)**2)
                    dk = sensors[i].fov * l2 + tol # max range delta
                    sobs_k = G[k]
                    for sob_k in sobs_k:
                        if (abs(sob_k.r-sob_c.r)<dk and abs(sob_k.r+sob_c.r)>dk and not 
                            any([similar_paths(sob_c, sob_prev, sob_k, sensors) for sob_prev in G[k+1]])):
                            sob_c.insert_blink(sob_k)
                            sob_k.insert_flink(sob_c)
                            Lp2+=1
                            Lt2+=1+len(sob_k.lkb)
                    k-=1
    return G, Lt1

def similar_paths(sob_i, sob_j, sob_k, sensors):# Check if 3 nodes should be on same path or not
    pth =0.01# TODO: How to set this automatically
    vth = 0.01
    obij = pr.get_pos_from_rd(sob_i.r, sob_j.r, sob_i.d, sob_j.d, sob_i.sid , sob_j.sid, sensors)
    objk = pr.get_pos_from_rd(sob_k.r, sob_j.r, sob_k.d, sob_j.d, sob_k.sid , sob_j.sid, sensors)
    if obij and objk:
        if ((obij.x-objk.x)**2+(obij.y-objk.y)**2<pth) and ((obij.vx-objk.vx)**2+(obij.vy-objk.vy)**2<vth):
            return True
    return False
    
def enum_graph_sigs(G, sensors, lite=False):
    # If no signatures observed then they should be intialized inside loop, 
    # here we assume atleast 1 track goes across all sensors
    s=len(sensors)-1
    Final_tracks = []
    Fin_track_len = 0
    while s>0: # s is sensor index of source
        for p, sobc in enumerate(G[s]):
            Nb = len(sobc.lkb)
            Nf = len(sobc.lkf)
            if (Nf == 0) & (Nb>0): # Stop track and store in final tracks
                sg = ob.SignatureTracks(sobc.r, sobc.d, s, sobc.g)# create new signature
                signatures = get_tracks(G, sobc, sg, sensors)
                if lite:
                    Fin_track_len+=1
                else:
                    Final_tracks.extend(signatures)# At sensor 0, Nb=0 for all, so all tracks will be saved

        s=s-1 # Move to previous sensor
#    for tracks in signatures:   
#        for track in tracks:
#            Final_tracks.append(track) 
    return Final_tracks, Fin_track_len

def get_BruteComplexity(G):
    Fin_track_len = 0
    Fin_edge_len = 0
    NodeComplexity = {}
    def get_NodeComplexity(sid, oid):
        if (sid,oid) in NodeComplexity:
            return NodeComplexity[(sid,oid)]
        else:
            N = 0
            for ndf in G[sid][oid].lkf:
                N+= 1 + get_NodeComplexity(ndf.sid, ndf.oid)
            NodeComplexity[(sid,oid)] = N
            return N

    s = len(G)-1
    while s>0: # s is sensor index of source
        for p, sobc in enumerate(G[s]):
            Fin_edge_len += get_NodeComplexity(s, p)
        s=s-1 # Move to previous sensor
    
    
        
#    for tracks in signatures:   
#        for track in tracks:
#            Final_tracks.append(track) 
    return Fin_edge_len, Fin_track_len

def get_tracks(G, nd, sig, sensors): # recursively extract tracks
    if nd.lkb:
        child_sigs = []
        for tnd in nd.lkb:
            new_sig = cp.copy(sig)
            new_sig.add_update3(tnd.r, tnd.d, tnd.g, tnd.sid, sensors)
            sig_pool = get_tracks(G, tnd, new_sig, sensors)
            child_sigs.extend(sig_pool)
        return child_sigs
    else:
        return [sig]

def get_Ntracks(nd): # recursively extract tracks from beginning
    N=0
    if nd.lkf:
        for tnd in nd.lkf:
            N +=1+ get_Ntracks(tnd)#*2
    return N
#        d_c = garda[i].d # Doppler of current
#        r_cp = garda[i-1].r # ranges from prev(below) sensor
#        d_cp = garda[i-1].d # ranges from prev(below) sensor
#        l1 = np.sqrt((sensors[i].x - sensors[i-1].x)**2+(sensors[i].y - sensors[i-1].y)**2) # sensor separation
#        d = sensors[i].fov * l1 + tol # max range delta
#        links = [] # [[] for l in range(len(r_c))] # Initialize List of Ns empty lists
#        for j,r in enumerate(r_c):
#            link_pr = [idx for idx,rcp in enumerate(r_cp) if abs(rcp-r)<d] #prev node indexs
#            
#            # to keep track going across sensors we link trac to atleast one obs
##            if not link_pr: link_pr = [abs(r_cp-r).argmin()] # NOTE: Point to nearest link if no other range is close, 
#            vx_asc = (r*d_c[j] - r_cp[link_pr]*d_cp[link_pr])/l1 # vx estimate bw j AND prev
#            if (l1p) & (i>1):
#                prunedid = [pid for pid, (vxasc, pnodeid) in enumerate(zip(vx_asc, link_pr))
#                if np.min(abs(ordered_links[i-1][pnodeid].vxa - vxasc)) < tol2/l1] # check common vxa with prev nodes childs
##                print(prunedid)
#                link_new = [link_pr[idx] for idx in prunedid]
#                vx_new = [vx_asc[idx] for idx in prunedid]
#                
#            else:
#                link_new = link_pr
#                vx_new = vx_asc
##                links.append(ob.link(link_pr, vx_asc))
#            if l2p:# level 2 pruning
#                link_cur = link_pr # indices of previous nodes associated with r_j
#                link_new = []
#                for bki in range(i-1):# backtrack: check bands with prev sensors
#                    l2 = np.sqrt((sensors[i].x - sensors[i-2-bki].x)**2+(sensors[i].y - sensors[i-2-bki].y)**2) # sensor separation
#                    dbk = sensors[i].fov * l2 + tol # max range delta
#                    for bkid, bknodeid in enumerate(link_cur):
#                        r_cb= garda[i-2-bki].r[ordered_links[i-bki-1][bknodeid]]
#                        d_cb= garda[i-2-bki].d[ordered_links[i-bki-1][bknodeid]]
#                        vx_bk = (r*d_c[j] - r_cb*d_cb)/l2 # vx estimate bw j and backtrack
#                        if l3p:
##                            print('sensor{}: r={},d={}, vx={}, vxj={}'.format(i, r_cb, d_cb,vx_bk,vx_asc[bkid]))
#                            ordered_links[i-bki-1][bknodeid] = [idx for idx,(rcb,vxbk) in enumerate(zip(r_cb,vx_bk))
#                            if (abs(rcb-r)<dbk) & (abs(vxbk-vx_asc[bkid])<tol2/l2)] #replace with valid idx 
#                        else:
#                            ordered_links[i-bki-1][bknodeid] = [idx for idx,rcb in enumerate(r_cb)
#                            if abs(rcb-r)<dbk] #replace with valid idx 
#                        set().union(link_new,ordered_links[i-bki-1][bknodeid]) #ranges in (i-bki-2) linked to r_j
#                    link_cur = link_new
#                    link_new=[]
#            links.append(ob.link(link_new, vx_new))
def get_g_thres(sig, scale, ag_pfa):
    if sig.N<3:
        return -np.inf # only 1 target cant make sense
    else:
        return scale[1]*chi2.isf(ag_pfa, 2*sig.N, loc=0, scale=1)# NOTE: Fudge factor 10
#        return np.sqrt(2*sig.N*2) # Roughly 2 sigma 95%
    
def get_l_thres(sig, scale, al_pfa):
    if sig.N<2:
        return -np.inf # only 1 target cant make sense
    else:
        return scale[0]*chi2.isf(al_pfa, 2*sig.N, loc=0, scale=1)# TODO: Wald statistic is normalized by CRB

def get_order(G, new_nd, target_nds, path, sensors, USE_EKF=False): # Slim version, Oct 2019
    if target_nds:
        target_nds_valid = list(filter(lambda x: ~x.visited, target_nds))
        if path.N<2: # Cannot calculate straigtness with 2 nodes
            return target_nds_valid, [np.inf for _ in target_nds_valid]
        else:
            g_cost=[]
            for tnd in target_nds_valid:
                if USE_EKF:
                    new_cost = path.get_newfit_error_ekf(sensors, tnd.r, tnd.d, tnd.g, tnd.sid)
                else:
                    new_cost = path.get_newfit_error(sensors, tnd.r, tnd.d, tnd.g, tnd.sid)
#                except ValueError as err:
#                    print(err.args)
#                    continue # Can print error happened     
                g_cost.append(new_cost) # use trace maybe
        srtind = np.argsort(g_cost)
        childs = [target_nds[ind] for ind in srtind]
        gcs = [g_cost[ind] for ind in srtind]
    else:
        childs=[]
        gcs = []
    return childs, gcs

def get_order2(G, new_nd, target_nds, path, sensors): # Heavy
    if target_nds:
        child_sigst = []
        g_cost=[]
        for tnd in target_nds:
            new_sig = cp.copy(path)
            try:
                new_sig.add_update3(tnd.r, tnd.d, tnd.g, tnd.sid, sensors)
            except ValueError as err:
                print(err.args)
                continue # Can print error happened                    
            if path.N<2: # Cannot calculate straigtness with 2 nodes
                g_cost.append(np.inf)
            else:
                g_cost.append(sum(new_sig.gc)) # use trace maybe
            child_sigst.append(new_sig)
        srtind = np.argsort(g_cost)
        childs = [target_nds[ind] for ind in srtind]
        child_sigs = [child_sigst[ind] for ind in srtind]
    else:
        childs=[]
        child_sigs=[]
    return childs, child_sigs

def Brute(G, nd, sig, sel_sigs, pid, sensors, cfgp, scale, minP): # recursive implementation
    childs = get_order(G, nd, nd.lkf, sig, sensors)
    # childs, child_sigs = get_order(G, nd, nd.lkf, cp.copy(sig), sensors)
    ag_pfa, al_pfa, rd_wt = cfgp['ag_pfa'],cfgp['al_pfa'],cfgp['rd_wt']
    L3 = 0
    for ndc in childs:# Compute costs for all neighbors
        if not ndc.visited:
            ndc_sig = cp.copy(sig)
            ndc_sig.add_update3(ndc.r, ndc.d, ndc.g, ndc.sid, sensors)
            pnext = cp.copy(pid)
            pnext.append(ndc.oid)
            L3+=Brute(G, ndc, ndc_sig, sel_sigs, pnext, sensors, cfgp, scale, minP)
    if not childs:# check for min cost(if node has no childs)
        if sig.N>=minP:# Only tolerate 1 miss
            l_cost, g_cost = mle.est_pathllr(sig, sensors, minP, rd_wt, False)#+wp*(len(sensors)-sig.N)    
            L3+=1
            # np.set_printoptions(precision=3)
#            print((pid, l_cost, round(get_l_thres(sig)), sig.gc))
            if l_cost < get_l_thres(sig, scale, al_pfa): # Based on CRLB
                sig.llr = l_cost
                sig.pid = pid
                sel_sigs.append(sig)# sig in this list should be updated whenever in nd is updated
        return L3
def Brute_iter(Gfix, sel_sigs, sensors, glen, cfgp): # recursive implementation
    G = cp.copy(Gfix)
    scale = cfgp['hscale']
    hN = cfgp['hN']
    minP = len(sensors)# Start with full length track search
    L3 = 0 # Counting edges visited
    for h in range(hN):
#        print('Graph has {} nodes.'.format(sum(len(g) for g in G)))
        sig_final =[]
        for i, sobs in enumerate(G):
            for pid, sobc in enumerate(sobs):
    #            print(G[ind].val)
                sig_origin = ob.SignatureTracks(sobc.r, sobc.d, i, sobc.g)# create new signature
                L3+=Brute(G, sobc, sig_origin, sig_final, [pid], sensors, cfgp, scale, minP)
        new_sigs = visit_selsigs(G, sig_final)
        sel_sigs.extend(new_sigs)
        G, stopping_cr = remove_scc(G, sensors)# Add skip connection
        glen.append(sum(len(g) for g in G))
        if stopping_cr:# Until path of length minP left in Graph
            break
        scale = scale*cfgp[incr]
        if glen[-1]-glen[-2]==0 and minP>=cfgp['Tlen']:
            minP-=1
    return glen, L3

def add_skipedge(G, sensors, lskp=0):# remove
    for i, sobs in enumerate(G):# NOTE: Should delete node from G instead of creating new G
        for pid, sobc in enumerate(sobs):
            # Add new skip connections (NEW!!)
            k=i-lskp-1 
            tol = -1e-10 # tolerance for range bands (-ve implies narrower FOV)
            if k>=0:
                l2 = np.sqrt((sensors[i].x - sensors[k].x)**2+(sensors[i].y - sensors[k].y)**2)
                dk = sensors[i].fov * l2 + tol # max range delta
                sobs_k = G[k]
                for sob_k in sobs_k:
                    if sob_k not in sobc.lkb:
                        if (abs(sob_k.r-sobc.r)<dk and abs(sob_k.r+sobc.r)>dk and not 
                            any([similar_paths(sobc, sob_prev, sob_k, sensors) for sob_prev in G[k+1]])):
                            sobc.insert_blink(sob_k)
                            sob_k.insert_flink(sobc)
    return G
        
def remove_scc(G, sensors, minP=2):# efficient in-place implementation
    flag = 0
    Ns = len(sensors)
    for i, sobs in enumerate(G):# NOTE: Should delete node from G instead of creating new G
        oidn = 0
        del_ind = []
        for pid, sobc in enumerate(sobs):
            if sobc.visited:
                for ndc in sobc.lkb:# TODO: Add skip connection here
                    ndc.lkf.remove(sobc)
                for ndc in sobc.lkf:
                    ndc.lkb.remove(sobc)
                del_ind.append(pid)
            else:
                G[i][pid].oid = oidn # Update observation order
                oidn +=1
        for did in reversed(del_ind):  G[i].pop(did)
        if 1: # Early Exit
            if len(G[i])==0: # Just stop here if need to save time
                flag+=1
                if flag>Ns-minP: return G, True # Empty sensors
            else: # Reset flag if obs found
                flag = 0
    return G, False

def remove_skipedge(G):# Removes skip edges
    for i, sobs in enumerate(G):
        for pid, sobc in enumerate(sobs):
#            print(len(sobc.lkf),end=' : ')
            for sobk in reversed(sobc.lkf):
                if abs(sobk.sid-sobc.sid)>1:
                    sobc.lkf.remove(sobk)
                    sobk.lkb.remove(sobc)
#            print(len(sobc.lkf))
    return G

def Relax2(Gfix, sel_sigs, sensors, glen, cfgp): # recursive implementation
    G = cp.copy(Gfix)
    scale = cfgp['hscale']
    hN = cfgp['hN']
    L3 = 0
    Ns = minP = len(sensors)
    hq = []
    for h in range(hN):
#        print('Graph has {} nodes.'.format(sum(len(g) for g in G)))
        for i in range(Ns-minP+1):
            sobs = G[i]
            for pid, sobc in enumerate(sobs):
    #            print(G[ind].val)
                sig_origin = ob.SignatureTracks(sobc.r, sobc.d, i, sobc.g)# create new signature
                L3+=DFS(G, sobc, sig_origin, sel_sigs, [pid], sensors, cfgp, minP, hq, scale, opt=[False,False,False] )
        G, stopping_cr = remove_scc(G, sensors)# Add skip connection
        glen.append(sum(len(g) for g in G))
        if stopping_cr:# Until path of length minP left in Graph
            break
        scale = scale*cfgp['incr']
        if glen[-1]-glen[-2]==0 and minP>=cfgp['Tlen']: # Might wanna use the heap here for speed.
            minP-=1
    return glen, L3


def Rellr(Gfix, sel_sigs, sensors, glen, cfgp): # recursive implementation
    G = cp.copy(Gfix)
    scale = cfgp['hscale']
    hN = cfgp['hN']
    L3 = 0
    Ns = minP = len(sensors)
    min_chain_leng = max(Ns - cfgp['rob'],2)
    hq = []
    lg_thres = np.array([[chi2.isf(cfgp['al_pfa'], 2*i, loc=0, scale=1) for i in range(1,Ns+1)],
                    [chi2.isf(cfgp['ag_pfa'], 2*i, loc=0, scale=1) for i in range(1,Ns+1)]])
    lg_thres[0,0]=-np.inf
    for i in range(2):
        lg_thres[1][i]=-np.inf
        
    for h in range(hN):
#        print('Graph has {} nodes.'.format(sum(len(g) for g in G)))
        lg_thres_t = lg_thres
        for _ in range(4):
            for i in range(Ns-minP+1):
                sobs = G[i]
                for pid, sobc in enumerate(sobs):
        #            print(G[ind].val)
                    sig_origin = ob.SignatureTracks(sobc.r, sobc.d, i, sobc.g)# create new signature
                    L3+=DFS(G, sobc, sig_origin, sel_sigs, [pid], sensors, cfgp, minP, hq, lg_thres_t, opt=[False,False,False] )
            G, stopping_cr = remove_scc(G, sensors)# Add skip connection
            glen.append(sum(len(g) for g in G))
            if stopping_cr:# Until path of length minP left in Graph
                break
            lg_thres_t=[lgt*sc for (lgt,sc) in zip(lg_thres_t, scale)] # Relax LLR thres outer loop
        if minP>=min_chain_leng: # Might wanna use the heap here for speed.
            minP-=1
        else:
            break
    return glen, L3

def _heap(Gfix, sel_sigs, sensors, glen, cfgp): # Heap variant for Relax, SPEKF
    G = cp.copy(Gfix)
    L3= np.zeros(2)
    Ns = minP = len(sensors)
    min_chain_leng = max(Ns - cfgp['rob'],2)
    hq = []
    lg_thres = np.array([[chi2.isf(cfgp['al_pfa'], 2*i, loc=0, scale=1) for i in range(1,Ns+1)],
                    [chi2.isf(cfgp['ag_pfa'], 2*i, loc=0, scale=1) for i in range(1,Ns+1)]])
    lg_thres[0,0]=-np.inf
    for i in range(2):
        lg_thres[1][i]=-np.inf
    for rho in range(1,Ns-min_chain_leng):
        G = add_skipedge(G, sensors, rho)     
    for i in range(Ns-min_chain_leng+1):
        sobs = G[i]
        for pid, sobc in enumerate(sobs):
#            print(G[ind].val)
            sig_origin = ob.SignatureTracks(sobc.r, sobc.d, i, sobc.g)# create new signature
            L3+=DFS(G, sobc, sig_origin, sel_sigs, [pid], sensors, cfgp, minP, hq, lg_thres, opt=[False,False,False] )
    G, stopping_cr = remove_scc(G, sensors, minP)# Add skip connection
    glen.append(sum(len(g) for g in G))
    # Using heap to get leftover targets
    if cfgp['mode'][-4:]=='heap':
        # Make dict of remaining nodes keyed by sensor id, r, d
        leftover = {} # collections.defaultdict(int)
        for i, g in enumerate(G):
#            print([(nd.oid, ndi) for ndi, nd in enumerate(g) if not nd.visited])# DEBUG
            for ndi, nd in enumerate(g):
                if not nd.visited:
                    leftover[(i,nd.r, nd.d)] = nd.oid
        for q in hq:
            pidt = [leftover[(si,ri, di)] for (si, ri, di) in zip(q[4].sindx,q[4].r, q[4].d) if (si, ri, di) in leftover]
            if len(pidt) == q[4].N:
    #            print(q[3],q[4].r, path_check(G, q[4],pidt)) # DEBUG
                if path_check(G, q[4],pidt) and q[4].N>=min_chain_leng:
                    sel_sigs.append(q[4])
                    for (si,pi) in zip(q[4].sindx,pidt):# Mark new ones as visited
                        G[si][pi].visited = True
                        G[si][pi].used = len(sel_sigs)-1
    return glen, L3

def Relax(Gfix, sel_sigs, sensors, glen, cfgp): # Slim version
    G = cp.copy(Gfix)
    scale = cfgp['hscale']
    hN = cfgp['hN']
    L3 = np.zeros(2); hc = 0
    Ns = minP = len(sensors)
    min_chain_leng = max(Ns - cfgp['rob'],2)
    hq = []
    lg_thres = np.array([[chi2.isf(cfgp['al_pfa'], 2*i, loc=0, scale=1) for i in range(1,Ns+1)],
                    [chi2.isf(cfgp['ag_pfa'], 2*i, loc=0, scale=1) for i in range(1,Ns+1)]])
    lg_thres[0,0]=-np.inf
    for i in range(2):
        lg_thres[1][i]=-np.inf
    while hc<hN and minP>=min_chain_leng:
#        print(lg_thres)
        for h in range(cfgp['rob']+1): # was Ns - cfgp['Tlen']+1, range(hN)
    #        print('Graph has {} nodes.'.format(sum(len(g) for g in G)))
    #        lg_thres = np.array([[scale[0]*chi2.isf(cfgp['al_pfa'], 2*i, loc=0, scale=1) for i in range(1,Ns+1)],
    #                    [scale[1]*chi2.isf(cfgp['ag_pfa'], 2*i, loc=0, scale=1) for i in range(1,Ns+1)]])
            
            for i in range(Ns-minP+1):
                sobs = G[i]
                for pid, sobc in enumerate(sobs):
        #            print(G[ind].val)
                    sig_origin = ob.SignatureTracks(sobc.r, sobc.d, i, sobc.g)# create new signature
                    L3+=DFS(G, sobc, sig_origin, sel_sigs, [pid], sensors, cfgp, minP, hq, lg_thres, opt=[False,False,False] )
            G, stopping_cr = remove_scc(G, sensors, minP)# Add skip connection
            glen.append(sum(len(g) for g in G))
            if stopping_cr:# Until path of length minP left in Graph
#                print('Graph Empty',[len(g) for g in G])
                break
            if 1: # Reduce minP inner loop
                minP-=1
                if minP<min_chain_leng: break
                G = add_skipedge(G, sensors, Ns-minP)# Only to be called when minP decrements
                hc+=1
        if stopping_cr: break # Only 1 iter for heap mode
        lg_thres=[lgt*sc for (lgt,sc) in zip(lg_thres, scale)] # Relax LLR thres outer loop
        if cfgp['mode'][-1]!='2': # Nominal mode
            G = remove_skipedge(G) # NOTE: reset skip edges
    # Using heap to get leftover targets
    # if cfgp['mode'][-4:]=='heap':
    #     # Make dict of remaining nodes keyed by sensor id, r, d
    #     print('Wrong mode! check config. ')
    return glen, L3

def DFS(G, nd, sig, sel_sigs, pid, sensors, cfgp, minP, hq, lg_thres, opt=[True,False,False]): # code cleanup
    if sig==None:
        return 0
    L3 = np.zeros(2) # Count edges visited
    ag_pfa, al_pfa, rd_wt = cfgp['ag_pfa'],cfgp['al_pfa'],cfgp['rd_wt']
    if not nd.visited:# Check if tracks could be joined
        childs, gcs = get_order(G, nd, nd.lkf, sig, sensors, cfgp['mode']=='NN')
        L3[0]+=len(childs) # If counting all edges, make 1
        for (ndc, gcc) in zip(childs, gcs):# Compute costs for all neighbors
            if not path_check(G, sig, pid): break # Added to stop DFS if parent is visited!
            if not ndc.visited:
                pnext = cp.copy(pid)
                pnext.append(ndc.oid)
                ndc_sig = cp.copy(sig)
                if ndc_sig.N>2 and cfgp['mode'][-1]!='4' and gcc>lg_thres[1][-1]: # SAGA/NN Algo
                    continue
                if cfgp['mode']=='NN':
                    ndc_sig.add_update_ekf(ndc.r, ndc.d, ndc.g, ndc.sid, sensors)
                else:
                    ndc_sig.add_update3(ndc.r, ndc.d, ndc.g, ndc.sid, sensors)
                if ndc_sig.N>2:
                    
                    if cfgp['mode'][-1]!='4':
                        g_cost = sum(ndc_sig.gc)
                        if g_cost>lg_thres[1][-1]:
                            continue
                    else: # Relax4 algorithm (Slower)
                        L3[1]+=1
                        l_cost, g_cost = mle.est_pathllr(ndc_sig, sensors, minP+2, rd_wt)
                        if l_cost>lg_thres[0][-1]: # Avoid going deeper as cost only increases
                            continue
                L3+=DFS(G, ndc, ndc_sig, sel_sigs, pnext, sensors, cfgp, minP, hq, lg_thres, opt)


        if not nd.visited:# check for min cost(if node not used)
            if path_check(G, sig, pid): # Check that no member of chain is already visited
                if sig.N>=minP and sig.gc is not None: # Atleast 3 elements
                    l_cost, g_cost = mle.est_pathllr(sig, sensors, minP+2, rd_wt);
                    L3[1]+=1 # If ONLY Counting paths, make 1, ELSE 0
#                    print(minP, l_cost, lg_thres[0][sig.N-1], g_cost, lg_thres[1][sig.N-1], pid ) #USE THIS TO DEBUG
                    deg_free = min(sig.N+len(sensors)-minP-1, len(sensors)-1)
                    if l_cost < lg_thres[0][deg_free] and abs(sum(sig.gc))<lg_thres[1][deg_free]: # Based on CRLB
                        sig.llr = l_cost
                        sig.pid = pid
                        sel_sigs.append(sig)
                        update_G(G, sig.sindx, pid, True, len(sel_sigs)-1)# Stores id of sig in sel_sigs
                else: # For Heap mode
                    if sig.N>=minP-1 and sig.gc is not None: # Atleast 3 elements
                        l_cost, g_cost = mle.est_pathllr(sig, sensors, minP+2, rd_wt);
                        L3[1]+=1 # If ONLY Counting paths, make 1, ELSE 0
    #                    print(minP, l_cost, lg_thres[0][sig.N-1], g_cost, lg_thres[1][sig.N-1], pid ) #USE THIS TO DEBUG
                        deg_free = min(sig.N+len(sensors)-minP-1, len(sensors)-1)
                        if sig.N>=minP and l_cost < lg_thres[0][deg_free] and abs(sum(sig.gc))<lg_thres[1][deg_free]: # Based on CRLB
                            sig.llr = l_cost
                            sig.pid = pid
                            sel_sigs.append(sig)
                            update_G(G, sig.sindx, pid, True, len(sel_sigs)-1)# Stores id of sig in sel_sigs
                        elif sig.N>=minP-1:
                            try:
                                Ns = len(sensors)
                                sig.pid = pid
                                heapq.heappush(hq, [2*(Ns-sig.N)+l_cost, sig.N, abs(sum(sig.gc)), sig.state_end.mean, sig])
                            except Exception as e:
                                print(e, end=' ')

    return L3

def DFS2(G, nd, sig, sel_sigs, pid, sensors, cfgp, minP, scale=[1e2,1e4], opt=[True,False,False]): # recursive implementation
    cand_sig =[]
    llr_min = np.inf
    L3 = 0 # Count edges visited
    ag_pfa, al_pfa, rd_wt = cfgp['ag_pfa'],cfgp['al_pfa'],cfgp['rd_wt']
    if nd.visited:# Check if tracks could be joined
        if opt[0]:# Choose best path acc to min lcost
            sig_used = sel_sigs[nd.used]
            nid = list(sig_used.sindx).index(nd.sid)
            if nid<sig_used.N-1:
                sig.add_sig(sig_used, nid+1, sensors)# create new signature
                llr_new, gc_new = mle.est_pathllr(sig, sensors, minP+2, rd_wt);L3+=1
                if llr_new < sig_used.llr and abs(sum(sig.gc))<abs(sum(sig_used.gc)) and sig.N>=minP:
                    if llr_new<llr_min:
                        sig.llr = llr_new
                        sig.pid = pid+sig_used.pid[nid+1:]
                        cand_sig = sig
                        repl_sigid = nd.used
                        repl_point = nid           
                        llr_min = llr_new
    else:
        childs, child_sigs = get_order(G, nd, nd.lkf, cp.copy(sig), sensors)
        L3+=len(childs) # If counting all edges, make 1
        for (ndc, ndc_sig) in zip(childs, child_sigs):# Compute costs for all neighbors
            if not path_check(G, sig, pid): break # Added to stop DFS if parent is visited!
            if not ndc.visited:
                pnext = cp.copy(pid)
                pnext.append(ndc.oid)
                L3+=DFS(G, ndc, ndc_sig, sel_sigs, pnext, sensors, cfgp, minP, scale, opt)
            elif opt[1]:# Skip connection ndc
                sig_used = sel_sigs[ndc.used]
                nid = list(sig_used.sindx).index(ndc.sid)
                if nid<sig_used.N-1:
                    ndc_sig.add_sig(sig_used, nid+1, sensors)# create new signature
                    llr_new, gc_new = mle.est_pathllr(ndc_sig, sensors, minP+2, rd_wt)
                    L3+=1 # InActivate
                    if llr_new < sig_used.llr and abs(sum(ndc_sig.gc))<abs(sum(sig_used.gc)) and ndc_sig.N>=minP:
                        if llr_new<llr_min:
                            ndc_sig.llr = llr_new
                            ndc_sig.pid = pid+sig_used.pid[nid+1:]
                            cand_sig = ndc_sig
                            repl_sigid = ndc.used
                            repl_point = nid           
                            llr_min = llr_new
        if cand_sig: # Check if node got visited by better track
            if nd.visited:
                if llr_min < sel_sigs[nd.used].llr:
                    # Erase existing track without nd
                    sig_temp = sel_sigs[nd.used]
                    sel_sigs[nd.used] = None
                    update_G(G, sig_temp.sindx,sig_temp.pid, False, None)# Mark new ones as visited
                else:
                    cand_sig = []

        if not nd.visited:# check for min cost(if node not used)
            if sig.N>=minP:
                l_cost, g_cost = mle.est_pathllr(sig, sensors, minP+2, rd_wt);
                L3+=1 # If ONLY Counting paths, make 1
#                print(l_cost, get_l_thres(sig), g_cost, get_g_thres(sig), pid )
                if l_cost < get_l_thres(sig, scale, al_pfa) and abs(sum(sig.gc))<get_g_thres(sig, scale, ag_pfa): # Based on CRLB
                    if opt[2]:# Backtrack to find best path using lkb along sig
                        if path_check(G, sig, pid):
                            sig_min = sig
                            pid_min = pid
                            sigb = ob.SignatureTracks(nd.r, nd.d, nd.sid)
                            max_g_cost = sum(sig.gc)
                            bk_l_cost, sig_back, pid_bk = DFSr(G, nd, sigb, max_g_cost, [nd.oid], sensors,wt)
                            # Mark nodes as visited
                            if bk_l_cost < l_cost and path_check(G, sig_back, pid_bk):# If backward path not found, means it was visited by other path
                                sig_min = sig_back
                                pid_min = pid_bk# traversed in rev order
                            update_G(G, sig_min.sindx, pid_min, True, sig)
                            sel_sigs.append(sig_min)# sig in this list should be updated whenever in nd is updated
                    else:# Just check path doesn't have already used nodes
                        if path_check(G, sig, pid):
                            sig.llr = l_cost
                            sig.pid = pid
                            sel_sigs.append(sig)
                            update_G(G, sig.sindx, pid, True, len(sel_sigs)-1)# Stores id of sig in sel_sigs
                            # sig in this list should be updated whenever in nd is updated
    if cand_sig:# Replace with the one which minimizes LLR
        # Add new track at nd
        sig_used = sel_sigs[repl_sigid]
        for iold in range(repl_point):# MArk old as free
            si=sig_used.sindx[iold]
            pi=sig_used.pid[iold]
            G[si][pi].visited = False
            G[si][pi].used = None
        update_G(G, cand_sig.sindx,cand_sig.pid, True, repl_sigid)# Mark new ones as visited
        sel_sigs[repl_sigid]=cand_sig
        print('State changed to x={}'.format(cand_sig.pid))
    return L3

def update_G(G, sindx, pid, vis, used):
    for (si,pi) in zip(sindx,pid):# Mark new ones as visited
        G[si][pi].visited = vis
        G[si][pi].used = used
    return

def path_check(G, sig, pid, allow_breaks=False):# Allows crossing paths
    flag = 0
    for (ob_id, cur_sid) in zip(pid, sig.sindx):
#        print(cur_sid, ob_id)
        if G[cur_sid][ob_id].visited:
            if not allow_breaks:
                return False
            flag+=1# Increment visited node count
        else: 
            flag = 0# Reset to 0 if unvisited node appears
        if flag>1:# IF 2 consecutive nodes in path were visited
            return False
    return True
def DFSr2(G, nd, sigb, max_g_cost, pidb, sensors, wt):# Picks overall min L-cost over unvisited
    wp=5
    g_costs = []
    sig_list = []
    pidn_list = []
    childs, child_sigs = get_order(G, nd, nd.lkb, cp.copy(sigb), sensors)
    
    for (ndc, ndc_sig) in zip(childs, child_sigs):# Compute costs for all neighbors
        if 1: #np.trace(ndc_sig.state_end.cov) <=max_g_cost:
            if not ndc.visited:
                pidb_next = cp.copy(pidb)
                pidb_next.append(ndc.oid)
                g_cost, sig_min, pidn = DFSr2(
                        G, ndc, ndc_sig, max_g_cost, pidb_next, sensors, wt)
                g_costs.append(g_cost)
                sig_list.append(sig_min)
                pidn_list.append(pidn)
    if g_costs:
        ind = np.argmin(g_costs)
        return g_costs[ind], sig_list[ind], pidn_list[ind]
    else:
        l_cost = mle.est_pathllr(sigb, sensors, wt, rd_wt)
        return l_cost+wp*(len(sensors)-sigb.N), sigb, pidb
    
def DFSr(G, nd, sigb, max_g_cost, pidb, sensors, wt):#Use G-cost to search over unvisited
    wp=5
    childs, child_sigs = get_order(G, nd, nd.lkb, cp.copy(sigb), sensors)
    exit_flag=False
    for (ndc, ndc_sig) in zip(childs, child_sigs):# Compute costs for all neighbors
        if exit_flag: #np.trace(ndc_sig.state_end.cov) <=max_g_cost:
            break
        if not ndc.visited:
            pidb_next = cp.copy(pidb)
            pidb_next.append(ndc.oid)
            g_cost, sig_min, pidn = DFSr(
                    G, ndc, ndc_sig, max_g_cost, pidb_next, sensors, wt)
            exit_flag = True
    if exit_flag:
        return g_cost, sig_min, pidn
    else:
        l_cost = mle.est_pathllr(sigb, sensors, wt, rd_wt)
        return l_cost+wp*(len(sensors)-sigb.N), sigb, pidb

def visit_selsigs(G, sel_sigs):# For Brute force method
    sig_final = sorted(sel_sigs, key=lambda x: x.llr )#mle.est_pathllr(x, sensors, wt)
    if 0:# Just pick min
        Nc = min([len(sig_final),max([len(Gs) for Gs in G])])
        sel_sigs = sig_final[0:Nc]
    else:# Mark Graph in ascending order
        sig_new=[]
        for sig in sig_final:
            if path_check(G, sig, sig.pid):
                sig_new.append(sig)
                for (si, pi) in zip(sig.sindx, sig.pid):
                    G[si][pi].visited = True
                    G[si][pi].used = sig
        sel_sigs = sig_new
    return sel_sigs

def get_minpaths(G, sensors, mode, cfgp):
    sel_sigs =[] # Note: wt includes the crb_min for range, doppler
    L3 = 0
    glen = [sum(len(g) for g in G)]
    dispatcher ={'DFS':DFS, 'Brute': Brute, 'NN': Relax, 'SAGA': Relax,'Rellr': Rellr, 'Brute_iter': Brute_iter}
    # if mode in ['DFS','Brute']:
    #     for i, sobs in enumerate(G):
    #         for pid, sobc in enumerate(sobs):
    #             if 1:#not sobc.visited:
    #                 sig_origin = ob.SignatureTracks(sobc.r, sobc.d, i, sobc.g)# create new signature
    #                 dispatcher[mode](G, sobc, sig_origin, sel_sigs, [pid], sensors, cfgp)
    # if mode=='DFS':# Run once again
    #     for i, sobs in enumerate(G):
    #         for pid, sobc in enumerate(sobs):
    #             if 1:#not sobc.visited:
    #                 sig_origin = ob.SignatureTracks(sobc.r, sobc.d, i, sobc.g)# create new signature
    #                 dispatcher[mode](G, sobc, sig_origin, sel_sigs, [pid], sensors, cfgp)
    #     sig_new=[]
    #     for sig in sel_sigs:
    #         if sig!=None:
    #             sig_new.append(sig)
    #     sel_sigs= sig_new
    # if mode =='Rellr':
    #     glen, L3 = dispatcher[mode[:5]](G, sel_sigs, sensors, glen, cfgp)
    # if mode=='SAGA' or mode=='NN':#Run with relaxed params
        # if mode[-4:]=='heap':
        #     glen, L3 = _heap(G, sel_sigs, sensors, glen, cfgp)
        # else:
    glen, L3 = dispatcher[mode](G, sel_sigs, sensors, glen, cfgp)
    # if mode=='Brute_iter':#Run with relaxed params
    #     glen, L3 = dispatcher[mode](G, sel_sigs, sensors, glen, cfgp)

    # if mode=='Brute':
    #     sel_sigs = visit_selsigs(G, sel_sigs)
    
    return sel_sigs, glen, L3

def get_rndsig(G, nd, sig_rnd, sel_sigs, pid, sensors):
    # generate random signature if no asociation can be done.
    minP=len(sensors)-1
    out = False
    if nd:
        if sig_rnd.N >= minP:
            l_cost, g_cost = mle.est_pathllr(sig_rnd, sensors, minP)#+wp*(len(sensors)-sig.N)    
            sig_rnd.llr = l_cost
            sig_rnd.pid = pid
            sel_sigs.append(sig_rnd)
            sel_sigs.append(sig_rnd)
            return True
        childs, child_sigs = get_order(G, nd, nd.lkf, cp.copy(sig_rnd), sensors)
        for (ndc, ndc_sig) in zip(childs, child_sigs):
            pnext = cp.copy(pid)
            pnext.append(ndc.oid)
            out = get_rndsig(G, ndc, ndc_sig, sel_sigs, pnext, sensors)
            if out:
                break
    return out
    
def add_sosi_to_G(G, Gnx, tracks, sensors):
    # Make signature from graph tracks
    # For elementary Max flow approach, MCF is better so use that instead
    sigs=[]
    for t in tracks:
        for n in t:
            sid, pid = Gnx.node[n]['ids']
            sobc = G[sid][pid]
            if n==t[0]:
                new_sig = ob.SignatureTracks(sobc.r, sobc.d, sid, sobc.g)# create new signature
            else:
                new_sig.add_update3(sobc.r, sobc.d, sobc.g, sid, sensors)
        sigs.append(new_sig)
    return sigs