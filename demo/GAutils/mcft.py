#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:29:07 2019
Min Cost Flow Association 
based on mcftracker(CVPR2008 paper) https://github.com/watanika/py-mcftracker
@author: anantgupta
"""

import math, collections
from ortools.graph import pywrapgraph
import sys, time

from GAutils import ml_est as mle
from GAutils import proc_est as pr
from GAutils import mcftools as tools
from GAutils import config as cfg
from GAutils import objects as ob


class MinCostFlowTracker:
	"""
		Object tracking based on data association via minimum cost flow algorithm
		L. Zhang et al.,
		"Global data association for multi-object tracking using network flows",
		CVPR 2008
	"""

	def __init__(self, detections, tags, min_thresh, P_enter, P_exit, beta):
		self._detections = detections
		self._min_thresh = min_thresh

		self.P_enter = P_enter
		self.P_exit = self.P_enter
		self.beta = beta

		self._id2name = tools.map_id2name(tags)
		self._name2id = tools.map_name2id(tags)
		self._id2node = tools.map_id2node(detections)
		self._node2id = tools.map_node2id(detections)
		self._fib_cache = {0: 0, 1: 1}
		self.L3 = 0

	def _fib(self, n):
		if n in self._fib_cache:
			return self._fib_cache[n]
		elif n > 1:
			return self._fib_cache.setdefault(n, self._fib(n - 1) + self._fib(n - 2))
		return n

	def _find_nearest_fib(self, num):
		for n in range(num):
			if num < self._fib(n):
				return (n - 1, self._fib(n - 1))
		return (num, self._fib(num))

	def _calc_cost_enter(self):
		# print(-math.log(self.P_enter))
		return -math.log(self.P_enter)
		# return -self._calc_cost_detection(self.beta)*len(self._detections)/2

	def _calc_cost_exit(self):
		# print(-math.log(self.P_exit))
		return -math.log(self.P_exit)
		# return -self._calc_cost_detection(self.beta)*len(self._detections)/2


	def _calc_cost_detection(self, beta):
		# print(math.log(beta / (1.0 - beta)))
		return math.log(beta / (1.0 - beta))

	def _calc_cost_link(self, rect1, rect2, sensors, garda, eps=1e-7):
#		prob_iou = tools.calc_overlap(rect1, rect2)
#		hist1 = tools.calc_HS_histogram(image1, rect1)
#		hist2 = tools.calc_HS_histogram(image2, rect2)
#		prob_color = 1.0 - tools.calc_bhattacharyya_distance(hist1, hist2)
#
#		prob_sim = prob_iou * prob_color
		sid = rect1[3] # Source sensor id
		did = rect2[3] # Dest sensor id
		sr = rect1[0]
		sd = rect1[1]
		dr = rect2[0]
		dd = rect2[1]
		   
		trg = pr.get_pos_from_rd(sr,dr,sd,dd,sid,did,sensors)
		if trg==None:
			return 1e7
		prob_joint = mle.est_prob_joint(trg, sensors, garda, cfg.rd_wt)
		# edge_llr = mle.est_edge_negllr(trg, sensors, garda, cfg.rd_wt)
		self.L3 += 1
#		print(prob_joint)
		return -math.log(prob_joint+1e-9)
		# return edge_llr

	def build_network(self, garda, sensors, h=0, f2i_factor=100):
		self.mcf = pywrapgraph.SimpleMinCostFlow()
		tol = 0.01
		for image_name, rects in sorted(self._detections.items(), key=lambda x: int(x[0][6:])):
			for i, rect in enumerate(rects):
				self.mcf.AddArcWithCapacityAndUnitCost(self._node2id["source"], self._node2id[(image_name, i, "u")], 1, int(self._calc_cost_enter() * f2i_factor))
				self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(image_name, i, "u")], self._node2id[(image_name, i, "v")], 1, int(self._calc_cost_detection(rect[4]) * f2i_factor))
				self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(image_name, i, "v")], self._node2id["sink"], 1, int(self._calc_cost_exit() * f2i_factor))
			frame_id = self._name2id[image_name]
			if frame_id == 0:
				continue
			for hi in range(min(h+1, frame_id)):
				prev_image_name = self._id2name[frame_id - 1 - hi]
				if prev_image_name not in self._detections:
					continue

				for i, i_rect in enumerate(self._detections[prev_image_name]):
					for j, j_rect in enumerate(rects):
						l1 = math.sqrt((sensors[i_rect[3]].x - sensors[j_rect[3]].x)**2+(sensors[i_rect[3]].y - sensors[j_rect[3]].y)**2) # sensor separation
						d = sensors[i_rect[3]].fov * l1 + tol # max range delta
						if abs(i_rect[0]-j_rect[0])<d and abs(i_rect[0]+j_rect[0])>d: # Only add valid links
							# print(prev_image_name, i, "v"', -> ' ,image_name, j, "u",' : ', int(self._calc_cost_link(i_rect, j_rect, sensors, garda) * f2i_factor))
							self.mcf.AddArcWithCapacityAndUnitCost(self._node2id[(prev_image_name, i, "v")], self._node2id[(image_name, j, "u")],
								1, int(self._calc_cost_link(i_rect, j_rect, sensors, garda) * f2i_factor))

	def _make_flow_dict(self):
		self.flow_dict = {}
		for i in range(self.mcf.NumArcs()):
			if self.mcf.Flow(i) > 0:
				tail = self.mcf.Tail(i)
				head = self.mcf.Head(i)
				if self._id2node[tail] in self.flow_dict:
					self.flow_dict[self._id2node[tail]][self._id2node[head]] = 1
				else:
					self.flow_dict[self._id2node[tail]] = {self._id2node[head]: 1}

	def _fibonacci_search(self, search_range=200):
		s = 0
		k_max, t = self._find_nearest_fib(self.mcf.NumNodes() // search_range)
		cost = {}

		for k in range(k_max, 1, -1):
			# s < u < v < t
			u = s + self._fib(k - 2)
			v = s + self._fib(k - 1)

			if u not in cost:
				self.mcf.SetNodeSupply(self._node2id["source"], u)
				self.mcf.SetNodeSupply(self._node2id["sink"], -u)

				if self.mcf.Solve() == self.mcf.OPTIMAL:
					cost[u] = self.mcf.OptimalCost()
				else:
					print("There was an issue with the min cost flow input.")
					sys.exit()

			if v not in cost:
				self.mcf.SetNodeSupply(self._node2id["source"], v)
				self.mcf.SetNodeSupply(self._node2id["sink"], -v)

				if self.mcf.Solve() == self.mcf.OPTIMAL:
					cost[v] = self.mcf.OptimalCost()
				else:
					print("There was an issue with the min cost flow input.")
					sys.exit()

			if cost[u] < cost[v]:
				t = v
			elif cost[u] == cost[v]:
				s = u
				t = v
			else:
				s = u

		self.mcf.SetNodeSupply(self._node2id["source"], s)
		self.mcf.SetNodeSupply(self._node2id["sink"], -s)

		if self.mcf.Solve() == self.mcf.OPTIMAL:
			optimal_cost = self.mcf.OptimalCost()
		else:
			print("There was an issue with the min cost flow input.")
			sys.exit()
		self._make_flow_dict()
		return (s, optimal_cost)

	def _brute_force(self, search_range=100):
		max_flow = self.mcf.NumNodes() // search_range
		# print("Search: 0 < num_flow <", max_flow)

		optimal_flow = 0
		optimal_cost = float("inf")
		for flow in range(max_flow):
			self.mcf.SetNodeSupply(self._node2id["source"], flow)
			self.mcf.SetNodeSupply(self._node2id["sink"], -flow)

			if self.mcf.Solve() == self.mcf.OPTIMAL:
				cost = self.mcf.OptimalCost()

				# print('Minimum cost:', self.mcf.OptimalCost())
				# print('')
				# print('  Arc    Flow / Capacity  Cost')
				# for i in range(self.mcf.NumArcs()):
				# 	cost = self.mcf.Flow(i) * self.mcf.UnitCost(i)
				# 	print('%1s -> %1s   %3s  / %3s       %3s' % (
				# 	self.mcf.Tail(i),
				# 	self.mcf.Head(i),
				# 	self.mcf.Flow(i),
			 #  		self.mcf.Capacity(i),cost))
			else:
				print("There was an issue with the min cost flow input.")
				sys.exit()

			if cost < optimal_cost:
				optimal_flow = flow
				optimal_cost = cost
				self._make_flow_dict()
		return (optimal_flow, optimal_cost)

	def run(self, fib=False, search_range=100):
		if fib:
			return self._fibonacci_search(search_range)
		else:
			return self._brute_force(search_range)

	# EXtract using 1 iteration ignoring flow length
def get_mcfsigs(garda, sensors, cfgp):
	
	Ns = len(sensors)
	# Parameters
	min_thresh = 0
	beta = 0.1 # math.exp(-1) # 0.2
	P_enter = math.exp(math.log(beta/(1-beta))*min(cfgp['rob'],Ns)/Ns/2) #0.1
	# print(math.exp(math.log(beta/(1-beta))*min(cfgp['rob'],Ns)/2) )#0.1
	P_exit = math.exp(math.log(beta/(1-beta))*min(cfgp['rob'],Ns)/Ns/2) #0.1
	fib_search = False # True

	# Prepare initial detecton results, ground truth, and images
	# You need to change below
	detections , tags , images = tools.create_tags(garda, sensors, beta)
	
	glen = [sum([len(gard.r) for gard in garda])]
	
	# Let's track them!
	start = time.time()
	tracker = MinCostFlowTracker(detections, tags, min_thresh, P_enter, P_exit, beta)
	tracker.build_network(garda, sensors)
	# print(tracker._calc_cost_enter())
	# print(tracker._calc_cost_exit())
	# print(tracker._calc_cost_detection(beta))
	optimal_flow, optimal_cost = tracker.run(fib=fib_search, search_range=len(sensors))
	if optimal_flow==0:
		optimal_flow, optimal_cost = tracker.run(fib=False, search_range=len(sensors))		
	end = time.time()
	if False:
		print("Finished: {} sec".format(end - start))
		print("Optimal number of flow: {}".format(optimal_flow))
		print("Optimal cost: {}".format(optimal_cost))

		# print("Optimal flow:")
		# print(tracker.flow_dict)
	sigs = []
	if optimal_flow>0:
		for st in tracker.flow_dict['source']:
			sid = int(st[0][6:])-1
			pid = st[1]
			pida = [st[1]]
			new_sig = ob.SignatureTracks(garda[sid].r[pid], garda[sid].d[pid], sid, garda[sid].g[pid])
			newt = list(tracker.flow_dict[list(tracker.flow_dict[st])[0]])[0]#Assuming only single link henceforth NOTE: This could handle more scenarios
			while newt!='sink':
				sid = int(newt[0][6:])-1
				pid = newt[1]
				pida.append(pid)
				new_sig.add_update3(garda[sid].r[pid], garda[sid].d[pid], garda[sid].g[pid], sid, sensors)
				newt = list(tracker.flow_dict[list(tracker.flow_dict[newt])[0]])[0]
			if new_sig.N>=max(2,Ns-cfgp['rob']):
				new_sig.pid = pida
				sigs.append(new_sig)
				# print(new_sig.state_end.mean)#DEBUG
	if not sigs:
		for sid, sensor in enumerate(sensors):
			if sid==0:
				sig_rnd = ob.SignatureTracks(math.sqrt(sensor.x**2+0.01), 0, sid, 1)
			else:
				sig_rnd.add_update3(math.sqrt(sensor.x**2+0.01), 0, 1, sid, sensors)
		sigs.append(sig_rnd)
		print('.',end='')
	V=tracker.mcf.NumNodes()
	E=tracker.mcf.NumArcs()
	glen.append(glen[0]-sum(sg.N for sg in sigs))
	L3a = int(V*E*math.log(V)) # Haque S.O.T.A. Slide 20
	return sigs, glen, [L3a, tracker.L3]

	# EXtract using Ns iteration ignoring flow length
def get_mcfsigs_all(garda, sensors, cfgp):
	# Prepare initial detecton results, ground truth, and images
	# You need to change below
	detections , tags , images = tools.create_tags(garda, sensors)

	# Parameters
	min_thresh = 0
	P_enter = 0.1
	P_exit = 0.1
	beta = 0.5
	fib_search = True
	glen = []
	Ns = minP = len(sensors)
	sigs = []
	L3a = 0
	min_chain_leng = max(Ns - cfgp['rob'],2)
	# Let's track them!
	start = time.time()
	garda_old = garda
	for h in range(Ns - min_chain_leng+1): #was Ns - min_chain_leng+1
		if 'tracker' in locals():
			del tracker
			detections , tags , images = tools.create_tags(garda, sensors)
		tracker = MinCostFlowTracker(detections, tags, min_thresh, P_enter, P_exit, beta)
		tracker.build_network(garda, sensors, h)
		optimal_flow, optimal_cost = tracker.run(fib=fib_search, search_range=len(sensors))
		end = time.time()
		if False:
			print("Finished: {} sec".format(end - start))
			print("Optimal number of flow: {}".format(optimal_flow))
			print("Optimal cost: {}".format(optimal_cost))

			print("Optimal flow:")
			print(tracker.flow_dict)
		
		if optimal_flow>0:
			for st in tracker.flow_dict['source']:
				sid = int(st[0][6:])-1
				pid = st[1]
				pida = [st[1]]
				new_sig = ob.SignatureTracks(garda[sid].r[pid], garda[sid].d[pid], sid, garda[sid].g[pid])
				newt = list(tracker.flow_dict[list(tracker.flow_dict[st])[0]])[0]#Assuming only single link henceforth NOTE: This could handle more scenarios
				while newt!='sink':
					sid = int(newt[0][6:])-1
					pid = newt[1]
					pida.append(pid)
					new_sig.add_update3(garda[sid].r[pid], garda[sid].d[pid], garda[sid].g[pid], sid, sensors)
					newt = list(tracker.flow_dict[list(tracker.flow_dict[newt])[0]])[0]
#				print(new_sig.N, Ns-h, new_sig.state_end.mean, new_sig.r) # DEBUG
				if new_sig.N>=max(Ns-h,2):
					new_sig.pid = pida
					sigs.append(new_sig)
		if len(sigs)>0: # Update detections (Pruning)
			# detections , tags , images = tools.create_tags_filt(garda, sensors, sigs)
			garda = reduce_gard(garda, sensors, sigs) # Update observations
		V=tracker.mcf.NumNodes()
		E=tracker.mcf.NumArcs()
		glen.append(V/2-1)
		L3a += int(V*E*math.log(V)) # Haque S.O.T.A. Slide 20
	# If nothing found
	if not sigs:
		for sid, sensor in enumerate(sensors):
			if sid==0:
				sig_rnd = ob.SignatureTracks(math.sqrt(sensor.x**2+0.01), 0, sid, 1)
			else:
				sig_rnd.add_update3(math.sqrt(sensor.x**2+0.01), 0, 1, sid, sensors)
		sigs.append(sig_rnd)
		print('.',end='')
	V=tracker.mcf.NumNodes()
	E=tracker.mcf.NumArcs()
	
	return sigs, glen, [L3a, tracker.L3]

def reduce_gard(garda, sensors, sigs):
	# Creates graph from all obs in garda excluding those used in sigs
	oid_seen = collections.defaultdict(list)
	for sig in sigs:
		for sid,r,d,g in zip(sig.sindx, sig.r, sig.d, sig.g):
#			if sid not in oid_seen:
#				oid_seen[sid] = [pid]
#			else:
			oid_seen[sid].append((r,d,g))
	new_garda=[]
	for si, gard in enumerate(garda):
		gard_new = ob.gardEst()
		L=len(gard.g)
		for ri,di,gi in zip(gard.r, gard.d, gard.g):
			if (ri,di,gi) not in oid_seen[si]:
				gard_new.add_Est(gi, 0, ri, di)# NOTE: Try replacing rect[4] with gard.g[oid]
		new_garda.append(gard_new)

	return new_garda