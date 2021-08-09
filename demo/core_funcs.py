# Functions used to perform association

def get_order(G, new_nd, target_nds, path, sensors, USE_EKF=False, upper_thres=np.inf): # Slim version, Oct 2019
    if target_nds:
        target_nds_valid = list(filter(lambda x: ~x.visited, target_nds))
        if path.N<2: # Cannot calculate straigtness with 2 nodes
            return target_nds_valid, [np.inf for _ in target_nds_valid], [None for _ in target_nds_valid]
        else:
            g_cost=[]
#             for tnd in target_nds_valid:
#                 if USE_EKF:
#                     new_cost = path.get_newfit_error_ekf(sensors, tnd.r, tnd.d, tnd.g, tnd.sid)
#                 else:
#                     new_cost = path.get_newfit_error(sensors, tnd.r, tnd.d, tnd.g, tnd.sid)
# #                except ValueError as err:
# #                    print(err.args)
# #                    continue # Can print error happened     
#                 g_cost.append(new_cost) # use trace maybe
#             srtind = np.argsort(g_cost)
#             childs = [target_nds[ind] for ind in srtind]
#             gcs = [g_cost[ind] for ind in srtind]
            # array processing
            tnd_dict =  collections.defaultdict(list)
            tnd_vec = []
            states_vec = []
            for tnd in target_nds_valid:
                tnd_dict[tnd.sid].append(tnd)
            for (tnd_sid, tnd_grp) in tnd_dict.items():
                if USE_EKF:
                    new_costs, new_states, valid_tnd_ids = path.get_newfit_error_nn(sensors, tnd_grp, tnd_sid, upper_thres)
                else:
                    new_costs, new_states, valid_tnd_ids = path.get_newfit_error_grp(sensors, tnd_grp, tnd_sid, upper_thres)
                g_cost.extend(new_costs) # use trace maybe
                tnd_vec.extend([tnd_grp[i] for i in valid_tnd_ids])
                states_vec.extend(new_states)
            # print(g_cost)
            srtind = np.argsort(g_cost)
            # print(srtind, valid_tnd_ids)
            childs = [tnd_vec[ind] for ind in srtind]
            gcs = [g_cost[ind] for ind in srtind]
            states_sorted = [states_vec[ind] for ind in srtind]
#               
    else:
        childs=[]
        gcs = []
        states_sorted = []
    return childs, gcs, states_sorted