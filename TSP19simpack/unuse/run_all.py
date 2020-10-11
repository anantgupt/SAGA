import os, subprocess

subprocess.call(['python', 'script_all4.py', '--mode', 'Relax','--N_avg','80'])
subprocess.call(['python', 'script_all4.py', '--mode', 'mle','--N_avg','80'])

subprocess.call(['python', 'script_all4.py', '--mode', 'Relax','--N_avg','80','--sep_th','1'])
subprocess.call(['python', 'script_all4.py', '--mode', 'mle','--N_avg','80','--sep_th','1'])

subprocess.call(['python', 'script_all4.py', '--mode', 'SPEKF','--N_avg','80','--sep_th','1'])
subprocess.call(['python', 'script_all4.py', '--mode', 'SPEKF-heap','--N_avg','80','--sep_th','1'])

subprocess.call(['python', 'script_all4.py', '--mode', 'mcf_all','--N_avg','80','--sep_th','1'])
subprocess.call(['python', 'script_all4.py', '--mode', 'mcf','--N_avg','80','--sep_th','1'])
