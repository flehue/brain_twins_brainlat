import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import sys;sys.path.append("../");import utils
from skimage.metrics import structural_similarity as ssim
import pickle
import warnings
warnings.filterwarnings('ignore')
import psutil
import os
import sys

rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
threads = int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1


##load csvs
diagnosis = sys.argv[1] ##might want to change
if diagnosis =="MCI":
    simfolder = f"../output/sweep_G_target_AD-opti_SC_extend/"
else:
    simfolder = f"../output/sweep_G_target_{diagnosis}-opti_SC_extend/"
empfolder = "../../matched_empirical_data/pablo_files/"

simdata_BOLD_omat = pd.read_csv(simfolder+"BOLD_omats.csv").drop_duplicates(["G","target"]).sort_values(["G","target"]).reset_index(drop=True)
BOLD_columns = [f"{n1}_{n2}" for n1 in range(90) for n2 in range(90) if n1<n2]
Gs,targets = np.sort(simdata_BOLD_omat["G"].unique()),np.sort(simdata_BOLD_omat["target"].unique())

emp_BOLD_omat = ###LOAD IT HOWEVER YOU WANT


par_cols = ["G","target"]
#utils functions
def reconstruct_symm(flattened_matrix,N=90,k=1,what_on_diag = 1):
    triu_indices = np.triu_indices(N,k=k)
    out = np.zeros((N,N))
    out[triu_indices] = flattened_matrix
    
    out = (out+out.T)
    if k ==1:
        out[np.diag_indices(N)] = what_on_diag
    return out

##want ssim? unleash this monster
# def get_ssim(flat_emp_mat,flat_sim_mat,what_on_diag = 1):
#     joint_dist = np.concatenate((flat_emp_mat,flat_sim_mat))
#     mini,maxi = np.min(joint_dist),np.max(joint_dist)
    
#     square_emp_mat = reconstruct_symm(flat_emp_mat,N=90,k=1,what_on_diag=what_on_diag)
#     square_sim_mat = reconstruct_symm(flat_sim_mat,N=90,k=1,what_on_diag=what_on_diag)
#     return ssim(square_emp_mat,square_sim_mat,data_range=maxi-mini)

def get_corr(flat_emp_mat,flat_sim_mat):
    return np.corrcoef(flat_emp_mat,flat_sim_mat)[0,1]
    
    


#%%functions that find the optimal 

def optimize_mat(flat_mat,simdata_mat):
    ##we take flat stuff
    to_apply = lambda row: get_corr(flat_mat,row[BOLD_columns].values)
    vals = simdata_mat.apply(to_apply,axis=1)
    ##if you want ssim gotta change that line and handle the diagonal (must be full of zeros)
    
    ##return
    idx_max = vals.idxmax()
    opti_val = vals[idx_max]
    G,target = simdata_mat.loc[idx_max][["G","target"]].values
    optimal_mat =  simdata_mat.loc[idx_max][BOLD_columns].values

    return opti_val,G,target,optimal_mat


##lone BOLD omat
opti_val,G,target,optimal_mat = optimize_mat(flat_BOLD_omat,simdata_BOLD_omat,what_on_diag=0)
print(f"G={G:.3f},target={target:.3f},corr={opti_val:.3f}")