import os
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.io import loadmat


sys.path.insert(0, "/data/workspaces/neuromodelling/rherzog/Brainlat/EEG")
import Omat_copula


infolder = "/data/workspaces/neuromodelling/rherzog/Brainlat/BOLD/TS_SDH"
outfolder = infolder
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 64))


def get_id(filename):
    return os.path.basename(filename).replace("_timeseries.mat", "")


def in_to_out(full_path, outfolder=outfolder):
    subject_id = get_id(full_path)
    savepath_fc = os.path.join(outfolder, f"{subject_id}_fc.npy")
    savepath_omat = os.path.join(outfolder, f"{subject_id}_omat.npy")
    return savepath_fc, savepath_omat


def fc_omat(filename, roi_as_col=True):
    data = loadmat(filename)["ts"]

    if not roi_as_col:
        data = data.T

    subdata = data[:, :90]
    fc = np.corrcoef(subdata.T)
    omat = Omat_copula.get_omat(subdata)
    return fc, omat


def process_file(full_path):
    if not full_path.endswith(".mat"):
        return

    savepath_fc, savepath_omat = in_to_out(full_path)
    if os.path.exists(savepath_fc) and os.path.exists(savepath_omat):
        print(f"Skipping {get_id(full_path)}")
        return

    fc, omat = fc_omat(full_path)
    print(savepath_fc, savepath_omat)
    np.save(savepath_fc, fc)
    np.save(savepath_omat, omat)


paths = sorted(
    os.path.join(infolder, filename)
    for filename in os.listdir(infolder)
    if filename.endswith(".mat")
)

Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_file)(path) for path in paths
)
