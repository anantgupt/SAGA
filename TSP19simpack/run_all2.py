import os, subprocess
selected_file = 'script_all5.py'
N_cpu = '1'
N_avg ='80'
subprocess.call(['python', selected_file, '--mode', 'Relax','--N_avg',N_avg,'--N_cpu',N_cpu])
subprocess.call(['python', selected_file, '--mode', 'mle','--N_avg',N_avg,'--N_cpu',N_cpu])

subprocess.call(['python', selected_file, '--mode', 'Relax','--N_avg',N_avg,'--sep_th','1','--N_cpu',N_cpu])
subprocess.call(['python', selected_file, '--mode', 'mle','--N_avg',N_avg,'--sep_th','1','--N_cpu',N_cpu])

subprocess.call(['python', selected_file, '--mode', 'SPEKF','--N_avg',N_avg,'--N_cpu',N_cpu])
subprocess.call(['python', selected_file, '--mode', 'SPEKF-heap','--N_avg',N_avg,'--N_cpu',N_cpu])

subprocess.call(['python', selected_file, '--mode', 'mcf_all','--N_avg',N_avg,'--sep_th','1','--N_cpu',N_cpu])
subprocess.call(['python', selected_file, '--mode', 'mcf','--N_avg',N_avg,'--sep_th','1','--N_cpu',N_cpu])
