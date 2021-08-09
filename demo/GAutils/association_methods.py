# -*- coding: utf-8 -*-
#import scipy as sp
import numpy as np
from GAutils import objects as ob
import copy as cp
from GAutils import proc_est as pr
from GAutils import ml_est as mle
from GAutils import config as cfg

def get_init_angle(sel_sensor,garda, sensors, Nr):
    # sel_sensor: index of sensor
    # garda: State of all sensors
    # Nr: Total detections across all sensors
    Ns = len(sensors)
    sel_range = garda[sel_sensor].r
    aij = np.zeros([len(sel_range),Nr - len(sel_range)])
    sensor_sep = np.zeros([Ns-1,1])
    sindx = np.setdiff1d(np.linspace(0,Ns-1,dtype='int16'),sel_sensor)
    for i, ri in enumerate(sel_range):
        count=0
        atemp=[]
        for j in range(Ns-1):
            gard = garda[sindx[j]]
            rj=gard.r
            sensor_sep[j]= sensors[sel_sensor].x - sensors[sindx[j]].x # should be euclidean distance if y~=0
            rj_size=len(rj)
            atemp.append(( rj**2 - ri**2 - sensor_sep[j]**2 )/(2*sensor_sep[j]*ri))
            count+=rj_size
        aij[i]=np.hstack(atemp)
    return aij
        
def get_Phantoms(sel_sensor, garda, sensors, Nr):
    Ns = len(sensors)
    sel_range = garda[sel_sensor].r
    sel_dop = garda[sel_sensor].d
    Phantoms = [] #[ob.PointTarget(0,0,0,0) for i in range(Nr - len(sel_range))] # 
    for i, (ri,di) in enumerate(zip(sel_range,sel_dop)):
        for j in range(sel_sensor+1,Ns):
            for (rj,dj) in zip(garda[j].r, garda[j].d):
                obj = pr.get_pos_from_rd(ri, rj, di, dj, sel_sensor , j, sensors)
                if obj:
                    Phantoms.append(obj)
#        else: # 
#            for j in range(sel_sensor+1,Ns):
#                gard = garda[j]
#                rj=gard.r
#                dj=gard.d
#                sensor_sep= abs(sensors[j].x - sensors[sel_sensor].x) # should be euclidean distance if y~=0
#                # coordinates along axis of ellipse with foci at sensors
#                x_r = (ri**2 - rj**2) / (2 * sensor_sep) #+ sensor_sep[j]/2
#                x_est = x_r + (sensors[sel_sensor].x + sensors[j].x)/2  # shift x for linear sensor array
#                vx = (ri * di - rj * dj) / sensor_sep
#                y_r2=(ri**2 + rj**2 - (sensor_sep**2)/2 - 2* (abs(x_r)**2))/2 # Square y
#                for k, (xr, xest, vxr, yr2) in enumerate(zip(x_r, x_est, vx, y_r2)):
#                    if yr2>0: # NOTE: Discard phantoms if y can't be calculated
#                        yr = np.sqrt( yr2 ) # y estimate common across all 
#                        vyr = (ri*di - vxr*xr) / yr
#                        Phantoms.append(ob.PointTarget(xest,yr,vxr,vyr))
    return Phantoms
    
def associate_garda2(garda, sensors): # only enumerates pairwise phantoms
    Ns = len(sensors)
    Phantoms = []
    for sel_sensor in range(Ns-1):
        sel_range = garda[sel_sensor].r
        sel_dop = garda[sel_sensor].d
        for i, (ri,di) in enumerate(zip(sel_range,sel_dop)):
            j=sel_sensor+1
            for (rj,dj) in zip(garda[j].r, garda[j].d):
                obj = pr.get_pos_from_rd(ri, rj, di, dj, sel_sensor , j, sensors)
                if obj:
                    Phantoms.append(obj)
    return Phantoms

def associate_garda(garda, sensors, rho=1e5): # Enumerates all possible pairs
#    Nr=sum([len(gard.r) for gard in garda])# Total observations
#    Phantoms = []
#    for i, gard in enumerate(garda):
#        angles =get_init_angle(i, garda, sensors, Nr)# Angles subtended at sensor i
#        gard.a = angles.flatten()
#        Phantoms.append(get_Phantoms(i, garda, sensors, Nr)) # Add phantoms seen at sensor i
#    return Phantoms
    Ns = len(sensors)
    Phantoms = []
    for sel_sensor in range(Ns-1):
        sel_range = garda[sel_sensor].r
        sel_dop = garda[sel_sensor].d
        for i, (ri,di) in enumerate(zip(sel_range,sel_dop)):
            for j in range(sel_sensor+1,min(Ns,sel_sensor+1+rho)):
                for (rj,dj) in zip(garda[j].r, garda[j].d):
                    obj = pr.get_pos_from_rd(ri, rj, di, dj, sel_sensor , j, sensors)
                    if obj:
                        Phantoms.append(obj)
#                sensor_sep= abs(sensors[j].x - sensors[sel_sensor].x) # should be euclidean distance if y~=0
#                # coordinates along axis of ellipse with foci at sensors
#                x_r = (ri**2 - rj**2) / (2 * sensor_sep) #+ sensor_sep[j]/2
#                x_est = x_r + (sensors[sel_sensor].x + sensors[j].x)/2  # shift x for linear sensor array
#                vx = (ri * di - rj * dj) / sensor_sep
#                y_r2=(ri**2 + rj**2 - (sensor_sep**2)/2 - 2* (abs(x_r)**2))/2 # Square y
#                for k, (xr, xest, vxr, yr2) in enumerate(zip(x_r, x_est, vx, y_r2)):
#                    if yr2>0: # NOTE: Discard phantoms if y can't be calculated
#                        yr = np.sqrt( yr2 ) # y estimate common across all 
#                        vyr = (ri*di - vxr*xr) / yr
#                        Phantoms.append(ob.PointTarget(xest,yr,vxr,vyr))
    return Phantoms
 

def band_prune(garda, sensors):
    Ns=len(sensors)
    tol = 0 # tolerance for range bands
    tol2 = 4 # normalized variance m^2/s
    l1p = 0 # Level 1 doppler pruning
    l2p = 0 # Level 2 range pruning 
    l3p = 0 # Level 3 doppler pruning
    ordered_links=[[] for _ in range(Ns)]#List of Ordered lists linking object indices at sensors
    forward_links =[[] for _ in range(Ns)]
    forward_links[Ns-1]=[ 0 for _ in range(len(garda[Ns-1].r))]
    for i in range(1,Ns):
        r_c = garda[i].r # ranges of current sensor
        d_c = garda[i].d # Doppler of current
        r_cp = garda[i-1].r # ranges from prev(below) sensor
        d_cp = garda[i-1].d # ranges from prev(below) sensor
        l1 = np.sqrt((sensors[i].x - sensors[i-1].x)**2+(sensors[i].y - sensors[i-1].y)**2) # sensor separation
        d = sensors[i].fov * l1 + tol # max range delta
        links = [] # [[] for l in range(len(r_c))] # Initialize List of Ns empty lists
        for j,r in enumerate(r_c):
            link_pr = [idx for idx,rcp in enumerate(r_cp) if abs(rcp-r)<d] #prev node indexs
            # to keep track going across sensors we link trac to atleast one obs
#            if not link_pr: link_pr = [abs(r_cp-r).argmin()] # NOTE: Point to nearest link if no other range is close, 
            vx_asc = (r*d_c[j] - r_cp[link_pr]*d_cp[link_pr])/l1 # vx estimate bw j AND prev
            if (l1p) & (i>1):
                prunedid = [pid for pid, (vxasc, pnodeid) in enumerate(zip(vx_asc, link_pr))
                if np.min(abs(ordered_links[i-1][pnodeid].vxa - vxasc)) < tol2/l1] # check common vxa with prev nodes childs
#                print(prunedid)
                link_new = [link_pr[idx] for idx in prunedid]
                vx_new = [vx_asc[idx] for idx in prunedid]
                
            else:
                link_new = link_pr
                vx_new = vx_asc
#                links.append(ob.link(link_pr, vx_asc))
            if l2p:# level 2 pruning
                link_cur = link_pr # indices of previous nodes associated with r_j
                link_new = []
                for bki in range(i-1):# backtrack: check bands with prev sensors
                    l2 = np.sqrt((sensors[i].x - sensors[i-2-bki].x)**2+(sensors[i].y - sensors[i-2-bki].y)**2) # sensor separation
                    dbk = sensors[i].fov * l2 + tol # max range delta
                    for bkid, bknodeid in enumerate(link_cur):
                        r_cb= garda[i-2-bki].r[ordered_links[i-bki-1][bknodeid]]
                        d_cb= garda[i-2-bki].d[ordered_links[i-bki-1][bknodeid]]
                        vx_bk = (r*d_c[j] - r_cb*d_cb)/l2 # vx estimate bw j and backtrack
                        if l3p:
#                            print('sensor{}: r={},d={}, vx={}, vxj={}'.format(i, r_cb, d_cb,vx_bk,vx_asc[bkid]))
                            ordered_links[i-bki-1][bknodeid] = [idx for idx,(rcb,vxbk) in enumerate(zip(r_cb,vx_bk))
                            if (abs(rcb-r)<dbk) & (abs(vxbk-vx_asc[bkid])<tol2/l2)] #replace with valid idx 
                        else:
                            ordered_links[i-bki-1][bknodeid] = [idx for idx,rcb in enumerate(r_cb)
                            if abs(rcb-r)<dbk] #replace with valid idx 
                        set().union(link_new,ordered_links[i-bki-1][bknodeid]) #ranges in (i-bki-2) linked to r_j
                    link_cur = link_new
                    link_new=[]
            links.append(ob.link(link_new, vx_new))
        fwl = [[] for _ in r_cp] # fwd links from prev obs
        for j, link in enumerate(links):
            for bidx in link.indx:
                fwl[bidx].append(j)
        for bidx in range(len(r_cp)):
            forward_links[i-1].append(len(fwl[bidx]))# Add idx of current sensor j to fwd link from bidx
        ordered_links[i]=links # 3D list
    return ordered_links, forward_links


def enumerate_raw_signatures(garda, ordered_links, forward_links, sensors):
    # If no signatures observed then they should be intialized inside loop, 
    # here we assume atleast 1 track goes across all sensors
    Ns = len(ordered_links)-1
    s = Ns
#    print('Enumerate')
#    signatures = [[ob.SignatureTracks(rs,ds)] for (rs,ds) in zip(garda[s].r,garda[s].d)] # List of signature objects  
#    [print(signature.r , signature.d ) for tracks in signatures for signature in tracks]# echo intial track
    Final_tracks = []
    signatures = []
    while s>0: # s is sensor index of source
        lij = abs ( sensors[s].x - sensors[s-1].x) # Distance between consecutive sensors
        lc = (sensors[s].x + sensors[s-1].x )/ 2 # Center location
        signatures_new = [[] for _ in range(len(garda[s-1].r))]
        for p, link in enumerate(ordered_links[s]): # p:Index of source obs
            Nb = len(link.indx) # number of links to prev sensor
            Nf = forward_links[s][p] # Fwd Tracks starting from current sensor
            if (Nb == 0) & (Nf>0): # Stop track and store in final tracks
                for track in signatures[p]: 
                    Final_tracks.append(track)# At sensor 0, Nb=0 for all, so all tracks will be saved
            else: # Propagate all tracks at obs p
                if Nf==0: # If no forward link from this obs, (Will initialize & handle new tracks)
                    if link: # And there is a link to prev obs
                        for ip in link.indx:# ip: Index of target obs
                            sg = ob.SignatureTracks(garda[s].r[p],garda[s].d[p], s)# create new signature
                            sg.add_update(garda[s-1].r[ip], garda[s-1].d[ip], s-1, sensors)# lij, lc
                            signatures_new[ip].append(sg)#Add a signature at index of asc with prev sensor
                else: # This loop will only start at 2nd iteration (@ Ns-2)
                    for ip in link.indx:# ip: Index of target obs
                        for track in signatures[p]:
                            sg2 = cp.copy(track) # NOTE: Consider cp.deepcopy
                            sg2.add_update(garda[s-1].r[ip], garda[s-1].d[ip], s-1, sensors)# lij, lc
                            if ip>=len(signatures_new):
                                print('why')
                            signatures_new[ip].append(sg2)
        signatures = signatures_new # Shift new tracks to current list
#        [print(signature.r , signature.state_end.mean) for tracks in signatures for signature in tracks]
        s=s-1 # Move to previous sensor

    for tracks in signatures:   
        for track in tracks:
            Final_tracks.append(track) 
    return Final_tracks, len(Final_tracks)

def enumerate_pruned_signatures(garda, ordered_links, forward_links, sensors, rd_wt):
    # If no signatures observed then they should be intialized inside loop, 
    # here we assume atleast 1 track goes across all sensors
    Ns = len(ordered_links)-1
    s = Ns
    Final_tracks = []
    signatures = []
    pruned_phantoms = []
    while s>0: # s is sensor index of source
        lij = abs ( sensors[s].x - sensors[s-1].x) # Distance between consecutive sensors
        lc = (sensors[s].x + sensors[s-1].x )/ 2 # Center location
        signatures_new = [[] for _ in range(len(garda[s-1].r))]
        for p, link in enumerate(ordered_links[s]): # p:Index of source obs
            Nb = len(link.indx) # number of links to prev sensor
            Nf = forward_links[s][p] # Fwd Tracks starting from current sensor
            if (Nb == 0) & (Nf>0): # Stop track and store in final tracks
                for track in signatures[p]: 
                    Final_tracks.append(track)# At sensor 0, Nb=0 for all, so all tracks will be saved
            else: # Propagate all tracks at obs p
                indx_new=[]
                if Nf==0: # If no forward link from this obs, (Will initialize & handle new tracks)
                    if link: # And there is a link to prev obs
                        for ip in link.indx:# ip: Index of target obs
                            trg = pr.get_pos_from_rd(garda[s-1].r[ip],garda[s].r[p],garda[s-1].d[ip],garda[s].d[p],s-1,s,sensors)
                            mle_llr = mle.est_llr(trg, sensors[s-1:s+1], garda[s-1:s+1], rd_wt) # NOTE: LLR only computed over sensor pair
                            gin = [garda[s-1].g[ip], garda[s].g[p]] # Gains of obs pair
                            sin = [s-1, s] # Indices of sensor pair
                            llr_thres = mle.compute_llr_thres2(sensors[s-1:s+1], garda[s-1:s+1], rd_wt)
#                            llr_thres = mle.compute_llr_thres3(trg, gin, sin, sensors, rd_wt) # threshold for this intersecting pair 
                            if mle_llr > llr_thres:
                                if cfg.debug_plots:
                                    import matplotlib.pyplot as plt; plt.figure(23)
                                    plt.quiver(trg.x, trg.y, trg.vx, trg.vy, color='b')
                                    plt.text(trg.x, trg.y, int(mle_llr), fontsize=7)
                                sg = ob.SignatureTracks(garda[s].r[p],garda[s].d[p], s)# create new signature
                                sg.add_update(garda[s-1].r[ip], garda[s-1].d[ip], s-1, sensors)# lij, lc
                                signatures_new[ip].append(sg)#Add a signature at index of asc with prev sensor
                                indx_new.append(ip) # Save indices for pruned graph
                                pruned_phantoms.append(trg)# Add pruned phantoms
                        link.indx = indx_new # Update links
                else: # This loop will only start at 2nd iteration (@ Ns-2)
                    Flag = True
                    for ip in link.indx:# ip: Index of target obs
                        trg = pr.get_pos_from_rd(garda[s-1].r[ip],garda[s].r[p],garda[s-1].d[ip],garda[s].d[p],s-1,s,sensors)
                        mle_llr = mle.est_llr(trg, sensors[s-1:s+1], garda[s-1:s+1], rd_wt)# NOTE: LLR only computed over sensor pair
                        gin = [garda[s-1].g[ip], garda[s].g[p]] # Gains of obs pair
                        sin = [s-1, s] # Indices of sensor pair
                        llr_thres = mle.compute_llr_thres2(sensors[s-1:s+1], garda[s-1:s+1], rd_wt)
#                        llr_thres = mle.compute_llr_thres3(trg, gin, sin, sensors, rd_wt) # threshold for this intersecting pair
                        if mle_llr > llr_thres:
                            pruned_phantoms.append(trg)# Add pruned phantoms
                            if cfg.debug_plots:
                                import matplotlib.pyplot as plt; plt.figure(23)
                                plt.quiver(trg.x, trg.y, trg.vx, trg.vy, color='r')
                                plt.text(trg.x, trg.y, int(mle_llr), fontsize=7)
                            indx_new.append(ip) # Save indices for pruned graph
                            Flag = False
                            if signatures[p]: # If signatures reach this point
                                for track in signatures[p]:
                                    sg2 = cp.deepcopy(track) # NOTE: Consider cp.deepcopy
                                    sg2.add_update(garda[s-1].r[ip], garda[s-1].d[ip], s-1, sensors)# lij, lc
                                    if ip>=len(signatures_new):
                                        print('why')
                                    signatures_new[ip].append(sg2)
                            else: # Start new track if previous tracks were cutoff
                                sg = ob.SignatureTracks(garda[s].r[p],garda[s].d[p], s)# create new signature
                                sg.add_update(garda[s-1].r[ip], garda[s-1].d[ip], s-1, sensors)# lij, lc
                                signatures_new[ip].append(sg)#Add a signature at index of asc with prev sensor
                        if Flag: # If track could not proceed add to list of signatures
                            for track in signatures[p]:
                                Final_tracks.append(track)# At sensor 0, Nb=0 for all, so all tracks will be saved
                    link.indx = indx_new # Update links
        signatures = signatures_new # Shift new tracks to current list
#        [print(signature.r , signature.state_end.mean) for tracks in signatures for signature in tracks]
        s=s-1 # Move to previous sensor

    for tracks in signatures:   
        for track in tracks:
            Final_tracks.append(track) 
    return Final_tracks, len(Final_tracks), ordered_links, pruned_phantoms

def distance(target, signature):
    d = abs(signature.state_end.mean - np.array([target.x,target.y,target.vx]))**2
    return d

def compute_asc_error(signatures, scene, asc_targets, sensors):
    Nt =len(scene) # Number of true targets
    Dx=np.zeros([asc_targets, Nt])
    Dy=np.zeros([asc_targets, Nt])
    Dvx=np.zeros([asc_targets, Nt])
    y_est = np.zeros(asc_targets)
    vy_est = np.zeros(asc_targets)
    for t, target in enumerate(scene):
        for s, signature in enumerate(signatures):
            Dx[s,t] = np.squeeze(abs(signature.state_end.mean[0] - target.x)**2)
            xsa = signature.state_end.mean[0] - [sensors[sid].x for sid in signature.sindx]
            ys_est2 = signature.r **2 - xsa **2 # squared y estimate at sensors
            if any(ys_est2 < 0):
                y_est[s] = 0
                vy_est[s] = 0
                Dy[s,t] = 1e9
            else:
                y_est[s] = np.sqrt(np.mean(ys_est2))
                vy_est[s] = np.mean(signature.r*signature.d - signature.state_end.mean[1]*xsa) / y_est[s] # Estimated using other estimates
            Dy[s,t] = np.squeeze(abs(y_est[s]- target.y)**2)
            Dvx[s,t] = np.squeeze(abs(signature.state_end.mean[1] - target.vx)**2)
    Dtotal = np.array(Dx+Dy+Dvx) # Find total error NOTE: Can weight states differently
    sig_indx = Dtotal.argmin(axis=0) # Finds signature with min error for every target 
    track_var=[]
    for si, sg in enumerate(signatures):
        crb = crbxy(sg.state_end.mean[0],y_est[si],sensors) 
        track_var.append( sg.state_end.cov[0,0] / crb[0,0] )# scaled based on crb, Finds total cov of tracks (sum of cov, can be weighted)
    t_indx = np.arange(Nt)
    St_er = np.zeros(3)
    KF_er = np.zeros(2)
    for (t,s) in zip(t_indx,sig_indx):
        St_er = St_er + [Dx[s,t],Dy[s,t],Dvx[s,t]]
        KF_er = KF_er + np.diag(signatures[s].state_end.cov) #NOTE: Fix this
    sig_indx_auto=(np.argsort(track_var)) # Index of signatures detected automatically
    Auto_er = np.zeros(3)
    for i in range(Nt):# count thru targets
        auto_idx = sig_indx_auto[i]# signature with min 
        auto_tidx = Dtotal[auto_idx,:].argmin() # index of associated target
        Auto_er = Auto_er + [Dx[auto_idx,auto_tidx],Dy[auto_idx,auto_tidx],Dvx[auto_idx,auto_tidx]] #error of auto asc
    return St_er/Nt, KF_er/Nt, Auto_er/Nt, sig_indx, sig_indx_auto, track_var, y_est, vy_est

def crbxy(x, y, sensors):
    FIM = np.zeros([2,2])
    for s in sensors:
        FIM += np.outer([x-s.x,y-s.y],[x-s.x,y-s.y])/((x-s.x)**2 + (y-s.y)**2)
    try:
        out = np.linalg.inv(FIM)
    except:
        out = np.diag(1/np.diag(FIM))
    return out

