# This file stores the common functions

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
    return G, Lp1