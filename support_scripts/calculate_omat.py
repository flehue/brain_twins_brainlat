import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
from scipy.special import comb
import itertools
import matplotlib.pyplot as plt

#load some data



#%% run

def gaussian_ent_biascorr(N, T):
    from scipy.special import psi
    """
    Computes the bias corrector for the entropy n estimator based on covariance
    matrix of gaussians
    
    Parameters:
    -----------
    N : int
        Number of dimensions
    T : int
        Sample size
        
    Returns:
    --------
    biascorr : float
        Bias corrector value
    """
    # Calculate psi terms
    psiterms = psi((T - np.arange(1, N+1)) / 2)
    
    # Calculate bias corrector
    biascorr = 0.5 * (N * np.log(2/(T-1)) + np.sum(psiterms))
    
    return biascorr

def multi_fc(data):
    """
    Computes multiple functional connectivity measures based on linear,
    non-linear, weighted and non-weighted dependencies measures.
    Pearson, WSDM, WSMI, GCMI, HoeffI, CMI(x,y,z), O-Info(x,y,z)
    
    Parameters:
    -----------
    data : numpy.ndarray
        T x N matrix where T is time points and N is number of variables
        
    Returns:
    --------
    wsdm : numpy.ndarray
        Weighted Symbolic Distance Matrix
    wsgc : numpy.ndarray
        Weighted Symbolic Gaussian Copula matrix
    ham_dist : numpy.ndarray
        Hamming distance matrix
    mimat : numpy.ndarray
        Mutual Information matrix
    hdmat : numpy.ndarray
        Hoeffding Distance matrix
    cmimat : numpy.ndarray
        Conditional Mutual Information matrix
    omat : numpy.ndarray
        O-Information matrix
    """
    # Get dimensions
    T, N = data.shape
    
    # Define entropy function
    ent_fun = lambda x, y: 0.5 * np.log((2 * np.pi * np.e) ** x * y)
    
    # Symbolization and Computing Hamming Distances
    a_ids = (data[0:T-2,:] < data[1:T-1,:]) & (data[1:T-1,:] < data[2:T,:])
    b_ids = (data[0:T-2,:] > data[1:T-1,:]) & (data[1:T-1,:] > data[2:T,:])
    
    symb_mat = np.zeros((T-2, N), dtype='U1')
    symb_mat[a_ids] = 'a'
    symb_mat[b_ids] = 'b'
    
    
    # Computation of copula
    sortid = np.argsort(data, axis=0)
    copdata = np.zeros_like(data)
    
    # Equivalent to MATLAB's [~, copdata] = sort(sortid, 1)
    for col in range(N):
        copdata[:, col] = np.argsort(sortid[:, col])+1
    
    # Normalization to have data in [0,1]
    copdata = copdata / (T + 1)
    
    
    bc2 = gaussian_ent_biascorr(2, T)
    bc1 = gaussian_ent_biascorr(1, T)
    bcN = gaussian_ent_biascorr(N, T)
    bcNmin1 = gaussian_ent_biascorr(N-1, T)
    bcNmin2 = gaussian_ent_biascorr(N-2, T)
    
    # Uniform data to gaussian data
    gaussian_data = norm.ppf(copdata)
    
    # Removing inf
    gaussian_data[np.isinf(gaussian_data)] = 0
    
    # GC covariance matrix
    gc_covmat = (gaussian_data.T @ gaussian_data) / (T - 1)
    
    # Linear indices of pairwise interactions
    k_ints = np.array(list(itertools.combinations(range(N), 2)))
    nints = len(k_ints)
    
    # Initialize matrices
    mimat = np.zeros((N, N))
    hdmat = np.zeros((N, N))
    cmimat = np.zeros((N, N))
    mi = np.zeros(nints)
    cmi = np.zeros(nints)
    hd = np.zeros(nints)
    
    # Preparing data
    detmv = np.linalg.det(gc_covmat)
    single_vars = np.diag(gc_covmat)
    var_ents = ent_fun(1, single_vars) - bc1
    sys_ent = ent_fun(N, detmv) - bcN  # total system entropy
    reg_id = np.arange(N)
    
    for i in range(nints):
        # Mutual Info
        idx1, idx2 = k_ints[i]
        thiscovmat = gc_covmat[np.ix_([idx1, idx2], [idx1, idx2])]
        this_detmv = np.linalg.det(thiscovmat)
        this_var_ents = var_ents[[idx1, idx2]]
        thissys_ent = ent_fun(2, this_detmv) - bc2
        
        mi[i] = np.sum(this_var_ents) - thissys_ent
        
        # Conditional Mutual Info
        sel_id = np.setdiff1d(reg_id, k_ints[i])
        sel_id1 = np.setdiff1d(reg_id, [idx1])
        sel_id2 = np.setdiff1d(reg_id, [idx2])
        
        xzent = ent_fun(N-1, np.linalg.det(gc_covmat[np.ix_(sel_id1, sel_id1)])) - bcNmin1
        yzent = ent_fun(N-1, np.linalg.det(gc_covmat[np.ix_(sel_id2, sel_id2)])) - bcNmin1
        zent = ent_fun(N-2, np.linalg.det(gc_covmat[np.ix_(sel_id, sel_id)])) - bcNmin2
        
        cmi[i] = xzent + yzent - sys_ent - zent
    
    # Fill matrices
    for i in range(nints):
        idx1, idx2 = k_ints[i]
        mimat[idx1, idx2] = mi[i]
        mimat[idx2, idx1] = mi[i]
        
        cmimat[idx1, idx2] = cmi[i]
        cmimat[idx2, idx1] = cmi[i]
        
        hdmat[idx1, idx2] = hd[i]
        hdmat[idx2, idx1] = hd[i]
    
    omat = mimat - cmimat
    
    return omat
    

# np.savetxt("omat_py.txt",omat_python)
#%%
if __name__ == "__main__":
    print(hola)