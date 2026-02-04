import DMF_ISP_numba_EandI as DMF  # Import a custom module for DMF simulation
import BOLDModel as BD       # Import a module for BOLD signal simulation
import numpy as np
from scipy import signal      # Import for signal processing
import os
from time import time as tm
from skimage.metrics import structural_similarity as ssim
import gc
import pickle
import pandas as pd 
import calculate_omat
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys

import warnings               # Import to suppress warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

rank = int(os.environ['SLURM_ARRAY_TASK_ID'])  # Current task ID
threads = int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1  # Total number of threads

# nmega = "9"#str(sys.argv[1])
nmega = str(np.sort(pd.read_csv("NMEGAS.csv").astype(int).values.flatten())[:-1][-(rank+1)])
epsilon = 0.04 #Convergence rate 


#########create output folder
folder = f"output/optimize_target_from_omat_epsilon{epsilon}/"
folder_plots = folder+"plots/"
folder_BOLD = folder + "BOLD_per_seed/"
masterfile_path = folder+"master_file.csv"
os.makedirs(folder_plots, exist_ok=True)
os.makedirs(folder_BOLD, exist_ok=True)
##create masterfile
target_columns = "\t".join([f"ROI_{n}" for n in range(90)])
if not os.path.isfile(masterfile_path):
    with open(masterfile_path, 'w') as f:
        line = "N_MEGA\tG_base\ttarget_base\ttime_taken\tmean_entropy_E\tmean_entropy_I\tmean_FC_emp\tmean_omat_emp\t"
        line += target_columns+"\n"
        f.write(line)

#LOAD DATA, all indexed by N_MEGA
#map NMEGAS to diagnosis
# with open("input/N_MEGA_to_diagnosis_pablo.pickle","rb") as f:
#     diagnoses = pickle.load(f)
# diagnosis = diagnoses[nmega]

#optimal parameters to start // cambiar para que lea la nueva tabla con
#  optimos y diagnosticos salen de la tabla
optimal_df = pd.read_csv("input/optimals_BOLD_corr_optiSC_extend.csv")[["N_MEGA","G_BOLD_omat","target_BOLD_omat"]]
optimal_df["N_MEGA"] = optimal_df["N_MEGA"].astype(str)

#empirical matrices and GM /// CAMBIAR ACA EL INPUTS DE DONDE SALEN LAS MATRICES
emp_omats = np.load("input/PABLO_omats.npz") # cargar desde FC_BOLD_bigtable
emp_FCs = np.load("input/PABLO_FCs.npz") # cargar desde FC_BOLD_bigtable
GMs = np.load("input/GM_data.npz") # // quitar o chequear que calcen los NMEGAS nuevoas
SCs = np.load("input/optimized_SC.npz")
diagnosis = tabla[tabla["N_MEGA"]==nmega]["diagnosis"] #/// 

if diagnosis =="MCI":
    SC = SCs["SC_AD"]
else:
    SC = SCs[f"SC_{diagnosis}"]
###help functions

#maximum range within two arrays
get_range = lambda array1,array2: np.max((array1.max(),array2.max()))-np.min((array1.min(),array2.min()))

#reconstruct matrix from its upper triangular
def reconstruct_symm(flattened_matrix,N=90,k=1,what_on_diag = 1):
    triu_indices = np.triu_indices(N,k=k)
    out = np.zeros((N,N))
    out[triu_indices] = flattened_matrix
    out = (out+out.T)
    if k ==1:
        out[np.diag_indices(N)] = what_on_diag
    return out

#calculate entropies per each channel
def entropies_per_channel(rates,t_as_col=False):
    ##########gamma entropy
    from scipy.stats import gamma as gamma_dist

    if t_as_col:
        rates = rates.T
    N = rates.shape[1]
    
    entropies = np.zeros(N)
    ##we iterate over rois
    for roi in range(N):
        a,loc,scale = gamma_dist.fit(data=rates[:,roi],floc=0) ###fit gamma
        entropies[roi] =gamma_dist.entropy(a,loc,scale) ##calculate and save gamma entropy
    return entropies

def plot(folder,nmega):
    plt.figure(1)
    plt.clf()
    plt.suptitle(f"target optimization according to OMAT node strength\nN_MEGA = {nmega}",weight="bold")

    plt.subplot(221) #correlation vs iterations
    plt.title("gof")
    plt.plot(corrs,label="corr to omat")
    plt.plot(corrs_FC,label="corr to FC")
    plt.plot(ssims,label="ssim to omat")
    maxcorr_it = np.argmax(corrs) #################################################HERE'S WHERE CORRELATION IS MAXIMIZED
    plt.vlines(maxcorr_it,corrs.max()-0.02,corrs.max()+0.02,color="red")
    plt.legend()


    plt.subplot(222) #optimal targets vs gray matter
    x = all_targets[:,maxcorr_it];y=GM
    slope,intercept,r,p,_ = linregress(x,y)
    plt.scatter(x,y)
    plt.plot(x,intercept+slope*x,label=f"r = {r:.4f}")
    plt.xlabel(f"target achieved at area (at optimal iteration = {maxcorr_it})")
    plt.ylabel("GM of the area")
    plt.legend()

    plt.subplot(223) #optimal targets vs actually attained rates
    x = all_targets[:,maxcorr_it]; y = all_mean_rates_E[:,maxcorr_it]
    slope,intercept,r,p,_ = linregress(x,y)
    plt.scatter(x,y)
    plt.plot(x,intercept+slope*x,label=f"r = {r:.4f}")
    plt.xlabel(f"target achieved at area (at optimal iteration = {maxcorr_it})")
    plt.ylabel("mean rate achieved at area")
    plt.legend()

    plt.subplot(224) #optimal targets vs entropy of rates
    x = all_targets[:,maxcorr_it]; y = all_entropies_E[:,maxcorr_it]
    slope,intercept,r,p,_ = linregress(x,y)
    plt.scatter(x,y)
    plt.plot(x,intercept+slope*x,label=f"rE, r = {r:.4f}")

    x = all_targets[:,maxcorr_it]; y = all_entropies_I[:,maxcorr_it]
    slope,intercept,r,p,_ = linregress(x,y)
    plt.scatter(x,y)
    plt.plot(x,intercept+slope*x,label=f"rI, r = {r:.4f}")

    plt.xlabel(f"target achieved at area (at optimal iteration = {maxcorr_it})")
    plt.ylabel("entropy of area's rate")
    plt.legend()

    plt.tight_layout()

    #save figure with N_MEGA name
    plt.savefig(folder+f"{nmega}")

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



def generate_mat(target_vector,seed):
    DMF.target = target_vector
    DMF.seed = seed       # Set the DMF model seed
    
    rates_E,rates_I, timeSim = DMF.Sim()
    rates_E = rates_E[cutfrom::downsample_rates, :]  
    rates_I = rates_I[cutfrom::downsample_rates, :]  

    # seconds/thousands*downsample to npoints
    converged_time = int(100/0.001/downsample_rates) 
    rates_to_consider_E = rates_E[-converged_time:,:]
    rates_to_consider_I = rates_I[-converged_time:,:]

    #rates observables
    entropies_E = entropies_per_channel(rates_to_consider_E)
    mean_rates_E = rates_to_consider_E.mean(axis=0)
    entropies_I = entropies_per_channel(rates_to_consider_I)
    mean_rates_I = rates_to_consider_I.mean(axis=0)

    dtt = 0.01  # Time step for BOLD simulation
    
    # Simulate BOLD signals from firing rates
    BOLD_signals = BD.Sim(rates_E, DMF.nnodes, dtt)
    del rates_E,rates_I
    BOLD_signals = BOLD_signals[cutfrom:, :][::100, :]  # Discard the first 12000 points, downsample further
       
    # Bandpass filter coefficients using a Bessel filter
    a0, b0 = signal.bessel(3, 2 * 1 * np.array([0.01, 0.1]), btype='bandpass')
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals, axis=0).astype(np.float32)  # Apply bandpass filtering

    sim_FC_BOLD = np.corrcoef(BOLD_filt.T)
    sim_omat_BOLD = calculate_omat.multi_fc(BOLD_filt)

    return sim_FC_BOLD,sim_omat_BOLD,mean_rates_E,mean_rates_I,entropies_E,entropies_I,BOLD_filt


############################################
nseeds = 20 #Number of FCs computed in each iterations, use same number of threads
iters = 100 #Number of iterations, optimal value depends on epsilon

folder_output = folder + "output_per_nmega/"
os.makedirs(folder_output, exist_ok=True)

#Array to save observables across iterations
all_targets = np.zeros((90,iters)) 
all_mean_rates_E = np.zeros((90,iters))
all_entropies_E = np.zeros((90,iters))
all_mean_rates_I = np.zeros((90,iters))
all_entropies_I = np.zeros((90,iters))
###observables
ssims = np.zeros(iters)
corrs = np.zeros(iters)
corrs_FC = np.zeros(iters) 



G,target = optimal_df[optimal_df["N_MEGA"]==nmega][["G_BOLD_omat","target_BOLD_omat"]].values.flatten()
#initialize target_vector
target_vector = target*np.ones(90)



##load required things
emp_FC_BOLD = reconstruct_symm(emp_FCs[str(nmega)],what_on_diag=1)
emp_omat_BOLD = reconstruct_symm(emp_omats[str(nmega)],what_on_diag=0)
mean_FC = emp_FC_BOLD.mean()
mean_omat = emp_omat_BOLD.mean()
emp_strengths = np.sum(emp_omat_BOLD,axis=1)
GM = GMs[str(nmega)]

DMF.G = G
DMF.SC = 0.2*SC

gof = -1000
now = tm()
for it in range(iters):
    all_targets[:,it] = np.copy(target_vector)

    
    
    ##things that average over seeds
    sim_omat_BOLD = np.zeros((90,90))
    mean_rates_E_vector = np.zeros(90)
    entropies_E_vector = np.zeros(90)
    mean_rates_I_vector = np.zeros(90)
    entropies_I_vector = np.zeros(90)
    
    
    BOLD_to_save = {"it":it}
    
    #main loop over seeds
    for seed in range(nseeds):
        ##time taken
        print(nmega,it,seed)
        sim_FC_BOLD_temp,sim_omat_BOLD_temp,mean_rates_E,mean_rates_I,entropies_E,entropies_I,BOLD_filt = generate_mat(target_vector,seed)
        sim_omat_BOLD += sim_omat_BOLD_temp
        BOLD_to_save[f"seed{seed}_array"] = BOLD_filt.astype(np.float32)
        
        mean_rates_E_vector += mean_rates_E
        entropies_E_vector += entropies_E
        mean_rates_I_vector += mean_rates_I
        entropies_I_vector += entropies_I

    sim_omat_BOLD /= nseeds             
    mean_rates_E_vector /= nseeds
    entropies_E_vector /= nseeds
    mean_rates_I_vector /= nseeds
    entropies_I_vector /= nseeds
        
    

    ###save observables
    simi = ssim(sim_omat_BOLD,emp_omat_BOLD,data_range=get_range(sim_omat_BOLD,emp_omat_BOLD))
    ssims[it] = simi
    corr = np.corrcoef(sim_omat_BOLD[np.triu_indices(90,k=1)],emp_omat_BOLD[np.triu_indices(90,k=1)])[0,1]
    corrs[it] = corr
    
    
    if corr > gof: ##we ovewrite the optimal BOLD if it is better, otherwise not
        np.savez_compressed(folder_BOLD+f"BOLD_optimal_nmega{nmega}.npz",
                            **BOLD_to_save)
        gof = corr
        
    #########
    all_mean_rates_E[:,it] = mean_rates_E_vector
    all_entropies_E[:,it] = entropies_E_vector
    all_mean_rates_I[:,it] = mean_rates_I_vector
    all_entropies_I[:,it] = entropies_I_vector
    
    #just check how much does the FC improve
    corr_FC = np.corrcoef(sim_FC_BOLD_temp[np.triu_indices(90,k=1)],emp_FC_BOLD[np.triu_indices(90,k=1)])[0,1]
    corrs_FC[it] = corr_FC

    # update target_vector
    sim_strengths = np.sum(sim_omat_BOLD,axis=1)

    
    target_vector += epsilon*(emp_strengths-sim_strengths)


    target_vector[target_vector<0] = 0 ###we don't allow for negative target values

time_taken = tm()-now
maxcorr_it = np.argmax(corrs)
optimal_targets = all_targets[:,maxcorr_it]
optimal_entropy_E = all_entropies_E[:,maxcorr_it].mean()
optimal_entropy_I = all_entropies_I[:,maxcorr_it].mean()

#save stuff
filename = folder_output + f"nmega{nmega}_output.npz"

np.savez_compressed(filename,
                    all_targets = all_targets,
                    optimal_targets = optimal_targets,
                    all_mean_rates_E = all_mean_rates_E,
                    all_entropies_E = all_entropies_E,
                    all_mean_rates_I = all_mean_rates_I,
                    all_entropies_I = all_entropies_I,
                    ssims = ssims,
                    corrs = corrs,
                    corrs_FC=corrs_FC)
#save plot
plot(folder_plots,nmega)


#%%
target_vals = "\t".join([f"{val}" for val in optimal_targets])
with open(masterfile_path, 'a') as f:
    line = f"{nmega}\t{G}\t{target}\t{time_taken}\t{optimal_entropy_E}\t{optimal_entropy_I}\t{mean_FC}\t{mean_omat}\t"
    line += target_vals+"\n"
    f.write(line)

#free memory, if possible
gc.collect()

#%%

plt.figure(100)
plt.clf()
plt.imshow(sim_FC_BOLD_temp,cmap="jet")
plt.colorbar()
plt.show()


