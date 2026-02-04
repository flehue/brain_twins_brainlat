# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:05:54 2024

@author: carlo
"""

import numpy as np
import DMF_ISP_numba as DMF  # Import a custom module for DMF simulation
import BOLDModel as BD       # Import a module for BOLD signal simulation
from scipy import signal      # Import for signal processing
import time                   # Import for timing the simulations
import warnings               # Import to suppress warnings
# import os                     # Import to handle environment variables and file paths
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output
import os
import itertools
import calculate_omat
from time import time as tm
import gc

rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
threads = int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1


target,struct = (3.4,"cn")   
folder = f"output/sweep_tau_decay_across_Gs_target={target:.2f}_{struct}-SC_converge_test/"
if not os.path.isdir(folder):
    os.makedirs(folder,exist_ok=True)

outfile = folder + f"master_rank{rank}.txt"
if not os.path.isdir(outfile):
    with open(outfile, 'w') as f:
        line = "G\ttau\tJdecay\tlogtau\tlogJdecay\tseed\ttime\tmean_rates\tmean_FC\n"
        f.write(line) 


# Simulation settings for DMF (Deco Mean Field) model
DMF.tmax = 720000  # Total simulation time in milliseconds
DMF.dt = 1         # Integration step in 

DMF.target = target        # Set the target firing rate
# DMF.G = G      # Set global coupling

# Load and process structural connectivity (SC) matrix
SC = np.load("../model_EEG/av_SC_aal90_cn_ad_dft.npz")[f"avmat_{struct}"]

np.fill_diagonal(SC, 0)  # Remove self-connections
DMF.SC = 0.2*SC.copy()
DMF.nnodes = len(DMF.SC)  # Set the number of nodes
DMF.update()  # Update the DMF model parameters

##treat rate
cutfrom = 12000
# Downsample the firing rates
downsample_rates = 10

init = time.time()  # Start timing

#default parameters
DMF.sigma = 0.01      # Set noise scaling factor

DMF.model_1 = 1       # Enable plasticity model
DMF.model_2 = 1       #enable decay model

def run_iter(G,tau,Jdecay,seed):
    DMF.seed = seed       # Set the DMF model seed

    DMF.G = G
    DMF.tau_p = tau       #1.54 Set synaptic time constant
    DMF.Jdecay = Jdecay    #484000 Set the synaptic decay rate
    
    now = tm()
    rates, timeSim = DMF.Sim()
    rates = rates[cutfrom::downsample_rates, :]  # Take every 10th time point

    dtt = 0.01  # Time step for BOLD simulation
    # Simulate BOLD signals from firing rates
    BOLD_signals = BD.Sim(rates, DMF.nnodes, dtt)
    BOLD_signals = BOLD_signals[cutfrom:, :][::100, :]  # Discard the first 12000 points, downsample further
       
    # Bandpass filter coefficients using a Bessel filter
    a0, b0 = signal.bessel(3, 2 * 1 * np.array([0.01, 0.1]), btype='bandpass') 
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals, axis=0)  # Apply bandpass filtering

    # Compute the functional connectivity (FC) matrix
    FC = np.corrcoef(BOLD_filt.T)
    # omat = calculate_omat.multi_fc(BOLD_filt)
    mean_FC = FC.mean()

    ##segundos/miles*downsample tirao a puntos
    converged_time = int(100/0.001/downsample_rates) 
    mean_rates = rates[-converged_time:,:].mean().mean()
    
    time = tm()-now

    return time,mean_rates,mean_FC

Gs = np.linspace(1,8,10,endpoint=False)
taus = np.logspace(-3,1,41,endpoint=True)
Jdecays = np.logspace(2,6,41,endpoint=True)
nseeds = 3
seeds = range(nseeds)
sims = list(itertools.product(Gs,taus, Jdecays,seeds))

for sim, (G,tau, Jdecay,seed) in enumerate(sims):
    if sim % threads == rank:
        
        line = f"{G:.3f}\t{tau:.4f}\t{Jdecay:.4f}\t{np.log10(tau):.4f}\t{np.log10(Jdecay):.4f}\t{seed}\t"
        
        time,mean_rates,mean_FC = run_iter(G,tau,Jdecay,seed)
        
        line += f"{time:.3f}\t{mean_rates:.3f}\t{mean_FC:.3f}\n"
        
        with open(outfile, 'a') as f:
            f.write(line)
            
        gc.collect()
        
        


  
#%% Run the DMF simulation


#%%
