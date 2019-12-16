# Calls plotting script gen_plots --f __ with all folders in current directory 
import os, subprocess
from tqdm import tqdm

for folder in tqdm(os.listdir(os.path.curdir)):
	if folder[:6]=='result':
		subprocess.call(['python', 'gen_plots.py', '--f', folder])

