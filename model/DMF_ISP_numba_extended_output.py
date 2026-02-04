# -*- coding: utf-8 -*-x    
"""
Created on Wed Sep 16 10:54:54 2020

@author: Carlos Coronel
"""

import numpy as np
from scipy import signal
from numba import float64, int32, vectorize,njit, jit
from numba.core.errors import NumbaPerformanceWarning
from scipy.io import loadmat
import warnings

import numpy as np

def matrix_multiply(A, B):
    """
    Perform matrix multiplication between two matrices A and B.
    This function assumes that the number of columns in A is equal to the number of rows in B.
    
    Parameters:
    A (numpy.ndarray): The first matrix.
    B (numpy.ndarray): The second matrix.
    
    Returns:
    numpy.ndarray: The result of the matrix multiplication.
    """
    # Get the dimensions of the input matrices
    n_rows_A, n_cols_A = A.shape
    n_rows_B, n_cols_B = B.shape
    
    # Ensure that the number of columns in A matches the number of rows in B
    if n_cols_A != n_rows_B:
        raise ValueError("Number of columns in A must be equal to the number of rows in B.")
    
    # Initialize the result matrix with zeros
    C = np.zeros((n_rows_A, n_cols_B))
    
    # Perform the matrix multiplication manually
    for i in range(n_rows_A):
        for j in range(n_cols_B):
            for k in range(n_cols_A):
                C[i, j] += A[i, k] * B[k, j]
    
    return C


warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#Network parameters
SC = np.random.uniform(size=(90,90))#np.mean(loadmat('SCmatrices88healthy.mat')['SCmatrices'], 0)
np.fill_diagonal(SC, 0)
nnodes = len(SC)

#Simulation parameters
tmax = 780 #time in seconds
dt = 0.1 #integration step in seconds
downsampling = 1000 #BOLD downsampling
downsampling_rates = 1 #Firing rates downsampling
seed = 0 #random seed

#Model parameters
gE, gI = 310, 615 #slope (gain factors) of excitatory and inhibitory, respectively, input-to-output functions
IthrE, IthrI = 0.403, 0.287 #thresholds current above which the firing rates increase linearly with the input currents
tauNMDA, tauGABA = 100, 10 #Gating decay time constants
gamma = 0.641 #control NMDA receptors gating time decay
dE, dI = 0.16, 0.087 #onstants determining the shape of the curvature of H around Ith
I0 = 0.382 #The overall effective external input
WE, WI = 1, 0.7 #scales the effective external input
W_plus = 1.4 #weight of recurrent excitation
sigma = 0.01 #Noise scaling factor
JNMDA = 0.15 #weights all excitatory synaptic couplings
G = 0 #Global coupling

#Synaptic plasticity parameters
target = 3 #target mean firing rate in Hz
tau_p = 1  #s time constant for plasticity
Jdecay = 495439 #s JGaba decay in plasticity equation


#Plasticity on/off
model_1 = 0

@jit(int32(int32),nopython=True)
#This function is just for setting the random seed
def set_seed(seed):
    np.random.seed(seed)
    return(seed)

# Input-to-output function (excitatory)
@vectorize([float64(float64,float64,float64,float64)],nopython=True)
def rE(IE,gE,IthrE,dE):
    return(gE * (IE - IthrE) / (1 - np.exp(-dE * gE * (IE - IthrE))))

#Input-to-output function (inhibitory)
@vectorize([float64(float64,float64,float64,float64)],nopython=True)
def rI(II,gI,IthrI,dI):
    return(gI * (II - IthrI) / (1 - np.exp(-dI * gI * (II - IthrI))))
    
#Mean Field Model
@njit
def mean_field(y,SC,params,target):
    
    SE, SI, JGABA = y
    G, WE, WI, W_plus, I0, JNMDA, tauNMDA, tauGABA, gamma, tau_p, Jdecay, gE, IthrE, dE, gI, IthrI, dI = params
    
    IE_t = WE * I0 + W_plus * JNMDA * SE + G * JNMDA * SC @ SE - JGABA * SI
    II_t = WI * I0 + JNMDA * SE - SI * 1
    
    rE_t = rE(IE_t,gE, IthrE, dE)
    rI_t = rI(II_t,gI, IthrI, dI)
    
    SE_dot = -SE / tauNMDA + (1 - SE) * gamma * rE_t / 1000
    SI_dot = -SI / tauGABA + rI_t / 1000
    JGABA_dot = (-JGABA / Jdecay * model_2 + rI_t / 1000 * (rE_t / 1000 - target / 1000) / tau_p) * model_1
       
    return np.vstack((SE_dot,SI_dot,JGABA_dot)), rE_t,rI_t

@njit
#Mean Field Model
def Noise(sigma):  
    SE_dot = sigma * np.random.normal(0,1,nnodes)
    SI_dot = sigma * np.random.normal(0,1,nnodes)
    JGABA_dot = sigma * np.random.normal(0,0,nnodes) #no noise here
    
    return(np.vstack((SE_dot,SI_dot,JGABA_dot)))  

# #This recompiles the model functions
def update():
    mean_field.recompile()
    Noise.recompile()


def Sim(verbose = False, return_rates = False):
    """
    Run a network simulation with the current parameter values.
    
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.
    
    return_rates : Boolean, optional
        If True, firing rates of excitatory populations were returned, at a sampling rate of (1 / dt / downsampling_rates)
        The default is False.    

    Raises
    ------
    ValueError
        An error raises if the dimensions of SC and the number of nodes
        do not match.

    Returns
    -------
    Y_t_rates: ndarray
        Firing rates for each node (only if return_rates = True)
    t : TYPE
        Values of time.

    """
    global SC, tmax, dt, seed, target
    
    #All parameters of the DMF model
    params = np.array([G, WE, WI, W_plus, I0, JNMDA, tauNMDA, tauGABA, gamma, tau_p, Jdecay, gE, IthrE, dE, gI, IthrI, dI])

    #Setting the random seed
    #It controls the noise and initial conditions
    set_seed(seed)
    np.random.seed(seed)
    
    if SC.shape[0] != SC.shape[1] or SC.shape[0] != nnodes:
        raise ValueError("check SC dimensions (",SC.shape,") and number of nodes (",nnodes,")")
    
    if SC.dtype is not np.dtype('float64'):
        try:
            SC = SC.astype(np.float64)
        except:
            raise TypeError("SC must be of numeric type, preferred float")
        
    #Simulation time
    Nsim = int(tmax / dt)
    timeSim = np.linspace(0,tmax,Nsim)
    #Time after downsampling
    
    #Initial conditions    
    neural_ic = np.ones((1,nnodes)) * np.array([1,1,1])[:,None] 
    neural_Var = neural_ic
    rE = np.zeros(nnodes)
    rI = np.zeros(nnodes)
    Y_t_rates = np.zeros((len(timeSim),nnodes)) #Matrix to save values (firing rates)
    Y_t_rates_I = np.zeros((len(timeSim),nnodes))
    
    for i in range(0,Nsim):
        Y_t_rates[i,:] = rE              
        Y_t_rates_I[i,:] = rI
        derivs, rE,rI = mean_field(neural_Var, SC, params, target)
        neural_Var += derivs * dt + Noise(sigma) * np.sqrt(dt)
                
    return Y_t_rates,Y_t_rates_I, timeSim
    
    
        
def ParamsNode():
    pardict={}
    for var in ('gE','gI','IthrE','IthrI','tauNMDA','tauGABA','gamma',
                'dE','dI','I0','WE','WI','W_plus','sigma','JNMDA',
                'target','tau_p'):
        pardict[var]=eval(var)

 