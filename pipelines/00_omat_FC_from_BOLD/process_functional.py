# Import necessary libraries
import numpy as np                      # For numerical operations
from scipy.io import loadmat           # To load MATLAB .mat files
import os                              # For file and directory manipulation
import Omat_copula                     # Custom module to compute OMAT (presumably a functional connectivity measure)

# Configuration parameters

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

# Input and output folders
infolder = "../../data/original/BOLD/timeseries_extracted/"      # Directory containing input time series
outfolder = "../../data/derived/Omats_FCs/"    # Directory where output will be saved

# Extract subject ID from filename
get_id = lambda filename: filename.split("/")[-1].split("_")[0]

# Function to convert input path into corresponding output paths
def in_to_out(full_path, outfolder=outfolder):
    idd = get_id(full_path)  # Extract subject ID

    # Rebuild relative subpath excluding the filename
    savepath = outfolder + "/" + "/".join(full_path.split("/")[1:-1])
    
    # Build full paths to save FC and OMAT matrices
    savepath_fc = savepath + "/" + idd + "_fc.npy"
    savepath_omat = savepath + "/" + idd + "_omat.npy"
    return savepath_fc, savepath_omat

# Function to compute correlation matrix (FC) and OMAT from a time series file
def fc_omat(filename, roi_as_col=True):
    data = loadmat(filename)["ts"]     # Load time series data from .mat file
    
    if not roi_as_col:
        data = data.T                  # Transpose if ROIs are not in columns

    subdata = data[:, :90]             # Select the first 90 ROIs or time series
    fc = np.corrcoef(subdata.T)        # Compute correlation matrix (FC)
    omat = Omat_copula.get_omat(subdata)  # Compute OMAT using custom copula-based method
    return fc, omat

# Apply processing to a single file and save outputs
def process_directory(full_path):
    if ".mat" in full_path:
        # try:
        fc, omat = fc_omat(full_path)                      # Compute FC and OMAT
        savepath_fc, savepath_omat = in_to_out(full_path)  # Get output file paths
        print(savepath_fc,savepath_omat)

        np.save(savepath_fc, fc)                           # Save FC matrix
        np.save(savepath_omat, omat)                       # Save OMAT matrix

# Recursively collect all file paths from the input folder
paths = [os.path.join(dirpath, filename) 
         for dirpath, dirnames, filenames in os.walk(infolder) 
         for filename in filenames]


# Process only the subset of files assigned to this thread/rank
for p, path in enumerate(paths):
    if p % threads == rank:
        process_directory(path)
