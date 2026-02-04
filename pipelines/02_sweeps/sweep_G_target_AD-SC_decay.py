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


diagnosis = "AD"
output_folder = f"output/sweep_G_target_{diagnosis}-opti_SC_extend/"
os.makedirs(output_folder, exist_ok=True)
outfile = output_folder + f"master_rank{rank}.txt"
if not os.path.isfile(outfile):
    with open(outfile, 'w') as f:
        line = "G\ttarget\tseed\ttime\tmean_rates\tmean_FC\tmean_omat\n"
        f.write(line)
    


# Simulation settings for DMF (Deco Mean Field) model
DMF.tmax = 720000  # Total simulation time in milliseconds
DMF.dt = 1         # Integration step in milliseconds

# Load and process structural connectivity (SC) matrix
SC = np.load(f"continue_200iterplus_from_optimize_SC_output_over_ALLopen/all_SCs_{diagnosis}.npy")[:,:,-1]

np.fill_diagonal(SC, 0)  # Remove self-connections
DMF.SC = 0.2*SC.copy()
DMF.nnodes = len(DMF.SC)  # Set the number of nodes
DMF.update()  # Update the DMF model parameters

##treat rate
cutfrom = 12000
# Downsample the firing rates
downsample_rates = 10

#%%

init = time.time()  # Start timing

#default parameters
DMF.sigma = 0.01      # Set noise scaling factor
DMF.tau_p = 1.5       #2 Set synaptic time constant
DMF.Jdecay = 400000    #484000 Set the synaptic decay rate
DMF.model_1 = 1       # Enable plasticity model
DMF.model_2 = 1       #enable decay model

def run_iter(G,target,seed):
    DMF.seed = seed       # Set the DMF model seed
    DMF.target = target        # Set the target firing rate
    DMF.G = G      # Set global coupling
    
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

    os.makedirs(f"{output_folder}BOLD_per_seed/seed_{seed}/", exist_ok=True)
    np.savez_compressed(f"{output_folder}BOLD_per_seed/seed_{seed}/G={G:.3f}_target={target:.3f}_seed={seed}_BOLD",
                        BOLD_filt.astype(np.float32))

    # Compute the functional connectivity (FC) matrix
    FC_BOLD = np.corrcoef(BOLD_filt.T)
    omat_BOLD = calculate_omat.multi_fc(BOLD_filt)
    mean_FC,mean_omat = FC_BOLD.mean(),omat_BOLD.mean()

    ##segundos/miles*downsample tirao a puntos
    converged_time = int(100/0.001/downsample_rates) 
    mean_rates = rates[-converged_time:,:].mean().mean()

    os.makedirs(f"{output_folder}all_per_seed/seed_{seed}/", exist_ok=True)
    np.savez_compressed(f"{output_folder}all_per_seed/seed_{seed}/G={G:.3f}_target={target:.3f}_seed={seed}_all",
                        FC_BOLD = FC_BOLD.astype(np.float32),
                        omat_BOLD = omat_BOLD.astype(np.float32),
                        mean_rates = mean_rates.astype(np.float32))
    
    time = tm()-now

    return time,mean_rates,mean_FC,mean_omat


Gs = np.linspace(1,2.5,51,endpoint=True)
targets = np.linspace(2,4,51,endpoint=True)
nseeds = 25;initial_seed=0
seeds = range(initial_seed,initial_seed+nseeds)
sims = list(itertools.product(Gs, targets,seeds))

for sim, (G, target,seed) in enumerate(sims):
    if sim % threads == rank:
        
        line = f"{G:.3f}\t{target:.3f}\t{seed}\t"
        
        time,mean_rates,mean_FC,mean_omat = run_iter(G,target,seed)
        
        line += f"{time:.3f}\t{mean_rates:.3f}\t{mean_FC:.3f}\t{mean_omat}\n"
        
        with open(outfile, 'a') as f:
            f.write(line)
            
        gc.collect()
        
        


  
#%% Run the DMF simulation


#%%
