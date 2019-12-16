# Automatically extracts Separation Threshold, Pmiss for all Algorithms and prints a Table
# Run this at directory containing all Algorithm runs e.g. Nov_set3
import os, subprocess
from tqdm import tqdm
import pandas, heapq

tbl = {}

for (path, folder, files) in os.walk('.'):
	for file in files:
		if file=='params.txt':
			target = os.path.join(path,file)
			fname = ([nm[7:] for nm in path.split('/') if nm[:3]=='res'])[0]#
			param = open(target, 'r')
			try:
				for par in param.readlines():
					if par[:6]=='Sep_th':
						sep_th = str(par[7:]).strip()
					if par[:4]=='mode':
						mode = par[5:].strip()
					if par[:5]=='Pmiss':
						pmiss = str(par[6:]).strip().split()[0]
				tbl[(mode, sep_th+', '+pmiss)] = fname
				# print('\x1b[1;33;40m Removed ',target,'\x1b[0m')
			except:
				print('\x1b[1;32;40m Delete Failed ',target,'\x1b[0m')
			param.close()
pda = pandas.DataFrame.from_dict(tbl, 'index')
# print(pda)

rows = list(set([ k[0] for k in tbl.keys()]))
columns = list(set([ k[1] for k in tbl.keys()]))
# print(rows)
# print(columns)
th = [[None for _ in (columns)] for _ in (rows)]
for (t, v) in tbl.items():
	th[rows.index(t[0])][columns.index(t[1])] = v
pda2 = pandas.DataFrame(th, rows, columns)
print(pda2.T)
# print(th)