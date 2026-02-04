# -*- coding: utf-8 -*-
"""
Created on Wen Jan 06 12:23:19 2021

@author: Carlos Coronel

Code for SC optimization based in the method proposed by Deco et al. (2019) [1]. First, the model was fitted to the
empirical Functionl Connectivity matrix using the original Structural Connectivity matrix. Then, Structural
Connectivity was updated iteratively employing the point-to-point difference between empirical and simulated 
Functional Connectivity matrices. Negative values within the optimized Structural Connectivity matrix were discarded.
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
from time import time as tm
from skimage.metrics import structural_similarity as ssim
import gc
import pickle
import sys

rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
threads = int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1
diagnosis = sys.argv[1]


##first we obtain what the homotopic entries are
columns = np.array([f"{n1}_{n2}" for n1 in range(90) for n2 in range(90) if n1 < n2])
def check_homotopic(columns):
    i1,i2 = [int(num) for num in columns.split("_")]
    return i2 == i1+1
homotopic_entries = [(i,col)[0] for i,col in enumerate(columns) if check_homotopic(col)]


#Folders
# diagnoses = ("CN","AD","FTD")
# folder = f"optimize_SC_output_over_ALLopen/"
folder = f"continue_200iterplus_from_optimize_SC_output_over_ALLopen/"
folder_FCs = folder + 'temp_FCs/'
os.makedirs(folder_FCs, exist_ok=True)


def get_uptri(matrix):
    entries = np.triu_indices(90,k=1)
    return matrix[entries]

def matrix_recon(flattened_matrix,N=90,k=1,what_on_diag = 1):
    triu_indices = np.triu_indices(N,k=k)
    out = np.zeros((N,N))
    out[triu_indices] = flattened_matrix
    
    out = (out+out.T)
    if k ==1:
        out[np.diag_indices(N)] = what_on_diag
    return out


mean_BOLD_FC = np.load("../matched_empirical_data/meanBOLDmats_open.npz")[f"{diagnosis}_FC"]
flat_mean_BOLD_FC = get_uptri(mean_BOLD_FC)
#Uppter triangular of the empirical Functional Connectivity matrix
# if diagnosis == "FTD":
#     original_struct = np.load("av_SC_aal90_cn_ad_dft.npz")[f"avmat_dft"]
# else:
    # original_struct = np.load("av_SC_aal90_cn_ad_dft.npz")[f"avmat_{diagnosis.lower()}"]
original_struct = np.load(f"optimize_SC_output_over_ALLopen/all_SCs_{diagnosis}.npy")[:,:,-1]


###################model parameters
DMF.nnodes = 90  # Set the number of nodes
# Simulation settings for DMF (Deco Mean Field) model
DMF.tmax = 720000  # Total simulation time in milliseconds
DMF.dt = 1         # Integration step in milliseconds
##treat rate
cutfrom = 12000
# Downsample the firing rates
downsample_rates = 10
#default parameters
DMF.sigma = 0.01      # Set noise scaling factor
DMF.tau_p = 1.5       #2 Set synaptic time constant
DMF.Jdecay = 400000    #484000 Set the synaptic decay rate
DMF.model_1 = 1       # Enable plasticity model
DMF.model_2 = 1       #enable decay model
DMF.target= 2.5
DMF.update()  # Update the DMF model parameters
DMF.G = 1.5
def generate_FC(SC,seed):
    DMF.seed = seed       # Set the DMF model seed
    DMF.SC = 0.2*SC
    
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
    FC_BOLD = np.corrcoef(BOLD_filt.T)

    return FC_BOLD



seeds = 25 #Number of FCs computed in each iterations, use same number of threads
iters = 200 #Number of iterations
all_SCs = np.zeros((90,90,iters)) #Array to save all SCs across iterations
ssims = np.zeros(iters)
corrs = np.zeros(iters)
epsilon = 0.01 #Convergence rate

tuples = [(it,seed) for it in range(iters) for seed in range(seeds)]
#Current copy of the SC matrix
SC = np.copy(original_struct)

for sim,(it,seed) in enumerate(tuples):
    if sim%threads==rank:    
        print(it,seed)

        ##########if first seed, save current iteration
        if seed ==0:
            all_SCs[:,:,it] = np.copy(SC)

        
        init = time.time()
        
        FCs_names = [f'FC-iter-{it}-seed-{s}_{diagnosis}.npy' for s in range(seeds)]
        
        
        FC_BOLD_temp = generate_FC(SC,seed)
        np.save(folder_FCs + f'FC-iter-{it}-seed-{seed}_{diagnosis}.npy', FC_BOLD_temp)
        
        all_ready = False
        while not all_ready:
            all_ready = True
            for s in range(seeds):
                filepath = folder_FCs + f'FC-iter-{it}-seed-{s}_{diagnosis}.npy'
                if not (os.path.exists(filepath) and os.path.getsize(filepath) > 100): ##bytes
                    all_ready = False
                    break
            time.sleep(2)

        # sumation = 0
        # while sumation < seeds:
        #     sumation = 0
        #     for k in range(seeds):
        #         print(k)
        #         sumation += np.sum(FCs_names[k] == np.array(os.listdir(folder_FCs)))
        #     time.sleep(1)
        
        FC_BOLD = np.zeros((90,90))
        
        for sxx in range(seeds):
            FC_BOLD += np.load(folder_FCs + f'FC-iter-{it}-seed-{sxx}_{diagnosis}.npy')   
        FC_BOLD /= seeds #Averaged FC matrix across random seeds
        
        if seed ==0:
            simi = ssim(FC_BOLD,mean_BOLD_FC,data_range=2)
            ssims[it] = simi
            corr = np.corrcoef(FC_BOLD[np.triu_indices(90,k=1)],mean_BOLD_FC[np.triu_indices(90,k=1)])[0,1]
            corrs[it] = corr


        dist_sim = get_uptri(FC_BOLD) #Upper triangular of the simulated FC matrix


        flat_SC = get_uptri(SC)
        #Updating the C matrix
        flat_SC[homotopic_entries] += epsilon*(flat_mean_BOLD_FC - dist_sim)[homotopic_entries]
        
       # C = thresholding(C,density) #Fixing density
        flat_SC[flat_SC < 0] = 0 #Discarding negative values
        
        SC = matrix_recon(flat_SC,what_on_diag=0)
    
        
        if (it == (iters - 1)) & (seed == 0):
            consecutive_euclideans_diagonal = np.linalg.norm(np.diff(all_SCs,axis=2)[np.triu_indices(90,k=1)][homotopic_entries]
                ,axis=0)

            np.save(folder+f'convergence_{diagnosis}.npy', consecutive_euclideans_diagonal)
            np.save(folder+f'all_SCs_{diagnosis}.npy', all_SCs)
            np.save(folder+f"simis_{diagnosis}.npy",ssims)
            np.save(folder+f"corrs_{diagnosis}.npy",corrs)
        print(time.time() - init)