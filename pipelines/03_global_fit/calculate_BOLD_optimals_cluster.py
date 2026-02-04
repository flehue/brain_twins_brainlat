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


diagnosis = sys.argv[1]
how = "ssim"


if diagnosis =="MCI":
    simfolder = f"../output/sweep_G_target_AD-opti_SC_extend/"
else:
    simfolder = f"../output/sweep_G_target_{diagnosis}-opti_SC_extend/"
empfolder = "../../matched_empirical_data/pablo_files/"
do=False ##whether or not to symmetrize  << à la deco >>


outfile = f"optimals_BOLD_{how}_{diagnosis}-SC_opti_extend.csv"
if not os.path.isfile(outfile):
    with open(outfile,"w") as f:
        line = "N_MEGA\t"
        line += "G_BOLD_FC\ttarget_BOLD_FC\tGoF_BOLD_FC\t"
        line += "G_BOLD_omat\ttarget_BOLD_omat\tGoF_BOLD_omat\t"
        line += "G_joint_BOLDs\ttarget_joint_BOLDs\tGoF_FC_joint_BOLDs\tGoF_omat_joint_BOLDs\n"
        f.write(line)

#%%
simdata_BOLD_FC = pd.read_csv(simfolder+"BOLD_FCs.csv").drop_duplicates(["G","target"]).sort_values(["G","target"])#.rename({"K":"G"},axis=1)
simdata_BOLD_omat = pd.read_csv(simfolder+"BOLD_omats.csv").drop_duplicates(["G","target"]).sort_values(["G","target"])#.rename({"K":"G"},axis=1)
# master = pd.read_csv(simfolder+"master_sweep.csv").drop_duplicates(["G","target"]).sort_values(["G","target"])#.groupby(["G","target"]).agg(np.nanmean).reset_index()
simdata_BOLD_FC = simdata_BOLD_FC.reset_index(drop=True)
simdata_BOLD_omat = simdata_BOLD_omat.reset_index(drop=True)
# master = master.reset_index(drop=True)

BOLD_columns = [f"{n1}_{n2}" for n1 in range(90) for n2 in range(90) if n1<n2]

Gs,targets = np.sort(simdata_BOLD_omat["G"].unique()),np.sort(simdata_BOLD_omat["target"].unique())


# halt
#%%load empirical individuals
emp_BOLD_omat = pd.read_csv(empfolder+"PABLO_omat_with_demo.csv")
emp_BOLD_FC = pd.read_csv(empfolder+"PABLO_FC_with_demo.csv")

emp_BOLD_omat = emp_BOLD_omat[emp_BOLD_omat["Diagnosis"]==diagnosis]
emp_BOLD_FC = emp_BOLD_FC[emp_BOLD_FC["Diagnosis"]==diagnosis]

par_cols = ["G","target"]
# halt
#%%utils functions
def reconstruct_symm(flattened_matrix,N=90,k=1,what_on_diag = 1):
    triu_indices = np.triu_indices(N,k=k)
    out = np.zeros((N,N))
    out[triu_indices] = flattened_matrix
    
    out = (out+out.T)
    if k ==1:
        out[np.diag_indices(N)] = what_on_diag
    return out

##ssim BOLD

def get_ssim(flat_emp_mat,flat_sim_mat,what_on_diag = 1):
    joint_dist = np.concatenate((flat_emp_mat,flat_sim_mat))
    mini,maxi = np.min(joint_dist),np.max(joint_dist)
    
    square_emp_mat = reconstruct_symm(flat_emp_mat,N=90,k=1,what_on_diag=what_on_diag)
    square_sim_mat = reconstruct_symm(flat_sim_mat,N=90,k=1,what_on_diag=what_on_diag)
    return ssim(square_emp_mat,square_sim_mat,data_range=maxi-mini)

def get_corr(flat_emp_mat,flat_sim_mat):
    return np.corrcoef(flat_emp_mat,flat_sim_mat)[0,1]
    
    


#%%functions that find the optimal 

def optimize_mat(flat_mat,simdata_mat,how=how,what_on_diag=1):
    ##we take flat stuff
    if how == "ssim":
        to_apply = lambda row: get_ssim(flat_mat,row[BOLD_columns].values,what_on_diag=what_on_diag)
    else:
        to_apply = lambda row: get_corr(flat_mat,row[BOLD_columns].values)
    vals = simdata_mat.apply(to_apply,axis=1)
    
    ##return
    idx_max = vals.idxmax()
    opti_val = vals[idx_max]
    G,target = simdata_mat.loc[idx_max][["G","target"]].values
    optimal_mat =  simdata_mat.loc[idx_max][BOLD_columns].values

    return opti_val,G,target,optimal_mat


def optimize_jointly(flat_BOLD_omat,flat_BOLD_FC,
                     simdata_BOLD_omat,simdata_BOLD_FC,how=how):
    if how =="ssim":
        to_apply_BOLD_omat = lambda row: get_ssim(flat_BOLD_omat,row[BOLD_columns].values,what_on_diag=0)
        to_apply_BOLD_FC = lambda row: get_ssim(flat_BOLD_FC,row[BOLD_columns].values,what_on_diag=1)
    else:
        to_apply_BOLD_omat = lambda row: get_corr(flat_BOLD_omat,row[BOLD_columns].values)
        to_apply_BOLD_FC = lambda row: get_corr(flat_BOLD_FC,row[BOLD_columns].values)
    
    vals_BOLD_omat = simdata_BOLD_omat.apply(to_apply_BOLD_omat,axis=1)
    vals_BOLD_FC = simdata_BOLD_FC.apply(to_apply_BOLD_FC,axis=1)
    
    
    summed = vals_BOLD_omat + vals_BOLD_FC
    
    idx_max = summed.idxmax()
    opti_val = summed[idx_max]
    opti_val_BOLD_omat,opti_val_BOLD_FC = vals_BOLD_omat[idx_max],vals_BOLD_FC[idx_max]
    G,target = simdata_BOLD_omat.loc[idx_max][["G","target"]].values
    
    optimal_sim_BOLD_omat =  simdata_BOLD_omat.loc[idx_max][BOLD_columns].values
    optimal_sim_BOLD_FC =  simdata_BOLD_FC.loc[idx_max][BOLD_columns].values

    return (opti_val,opti_val_BOLD_omat,opti_val_BOLD_FC,
            G,target,optimal_sim_BOLD_omat,optimal_sim_BOLD_FC)

#############main loop
nmegas = emp_BOLD_omat["N_MEGA"].values.astype(int)

sims = nmegas

# results_dic = {}
for sim,nmega in enumerate(nmegas):
    if sim % threads == rank:
        print(nmega)
        
        line = f"{nmega}\t"
        
        flat_BOLD_FC = emp_BOLD_FC[emp_BOLD_FC["N_MEGA"] == nmega][BOLD_columns].values.flatten()
        flat_BOLD_omat = emp_BOLD_omat[emp_BOLD_omat["N_MEGA"] == nmega][BOLD_columns].values.flatten()
        
        ##lone BOLD FC
        opti_val,G,target,optimal_mat = optimize_mat(flat_BOLD_FC,simdata_BOLD_FC,what_on_diag=1)
        line += f"{G:.3f}\t{target:.3f}\t{opti_val:.4}\t"
        
        ##lone BOLD omat
        opti_val,G,target,optimal_mat = optimize_mat(flat_BOLD_omat,simdata_BOLD_omat,what_on_diag=0)
        line += f"{G:.3f}\t{target:.3f}\t{opti_val:.4}\t"
        
        ##jointly BOLDs
        opti_val,opti_val_BOLD_omat,opti_val_BOLD_FC,G,target,optimal_sim_BOLD_omat,optimal_sim_BOLD_FC = optimize_jointly(flat_BOLD_omat,flat_BOLD_FC,
                          simdata_BOLD_omat,simdata_BOLD_FC)
        line += f"{G:.3f}\t{target:.3f}\t{opti_val_BOLD_FC:.4}\t{opti_val_BOLD_omat:.4}\n"
    
    
        with open(outfile,"a") as f:
            f.write(line)