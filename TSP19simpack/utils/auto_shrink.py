# Auto deletes large pickle files to save space
import os, subprocess
from tqdm import tqdm

for (path, folder, files) in os.walk('.'):
	for file in files:
		if file in ['plot5.pickle','plot8.pickle','plot9.pickle']:
			target = os.path.join(path,file)
			try:
				os.remove(target)
				print('\x1b[1;33;40m Removed ',target,'\x1b[0m')
			except:
				print('\x1b[1;32;40m Delete Failed ',target,'\x1b[0m')
	# if file =='plot5.pickle':
	# 	print(path,' : ',folder)
	# print(path, ' : ',folder, ':',file)

