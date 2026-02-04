import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data1_omat = np.load("PABLO_omats.npz")
data2_omat = np.load("unmatched_omats.npz",allow_pickle=True)

data1_fc = np.load("PABLO_FCs.npz")
data2_fc = np.load("unmatched_FCs.npz",allow_pickle=True)

#%%
out_omat = {}
out_fc = {}

#omats
for key1 in data1_omat.files:
    omat = data1_omat[key1].astype(np.float32)
    out_omat[key1] = omat
    
for key2 in data2_omat.files:
    omat = data2_omat[key2].astype(np.float32)
    out_omat[key2] = omat
    
    
#omats
for key1 in data1_fc.files:
    fc = data1_fc[key1].astype(np.float32)
    out_fc[key1] = fc
    
for key2 in data2_fc.files:
    fc = data2_fc[key2].astype(np.float32)
    out_fc[key2] = fc

#%%
# np.savez_compressed("all_Omats.npz",**out_omat)
np.savez_compressed("all_FCs.npz",**out_omat)

# data3 = np.load("Omats_NMEGA.npz")
