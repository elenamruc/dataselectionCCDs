"""
Original code written by Nuria.
"""

import numpy as np
from numba import njit, prange
import pandas as pd
#from numba import config
#config.DEBUG_JIT = False


# UNITS
um = 1.0
cm = 10000*um
eV = 1.0

# PARAMETER FOR THE DIFFUSION MODEL
DIFF_A      = 803.5*um*um
DIFF_b      = 6.5e-4/um
DIFF_alpha  = 0.83
DIFF_beta   = 0.0112/eV

# CCD GEOMETRY
PIX_THICKNESS = 15.0*um
#ROWS = (0*um,1044*PIX_THICKNESS*um)
#COLS = (0*um,6144*PIX_THICKNESS*um)
ROWS = (0*um,6080*100*PIX_THICKNESS*um)
COLS = (0*um,6144*PIX_THICKNESS*um)
DEPTH= (0*um,670*um)

MEAN_Eeh_CREATION = 3.74*eV
FANOFACTOR = 0.12

# FOR CHARGE IONIZATION AND DIFFUSION
CHARGE_CREATION_PROBABILITITES = np.load("p100K.npz")['data'].T
CHARGE_CREATION_PROBABILITITES = {
        "E":CHARGE_CREATION_PROBABILITITES[0,:],
        "Prob":CHARGE_CREATION_PROBABILITITES[1:,:]
        }
        

    

################################################################################ CHARGE IONIZATION
def charge_ionization(E):
    """Number of electron-hole pair that will be created for a given ionization energy

    Probability distribution 

    E total energy of the event in units of eV

    """
    
    E = np.round(E)
    return E



################################################################################ DIFFUSION
@njit
def diffuse_event(z,Ee):
    """Diffuse the event by following the model

        sigma_xy^2(z,Ee) = -A log|1 - bz| (alpha - beta E_e)

    Units:
        A in units of um^{2}
        b in units of um^{-1}
        alpha unitless
        beta in units of eVee^{-1} 

    """
    
    sigma_xy = np.sqrt(-DIFF_A * np.log(1-DIFF_b*z)) * (DIFF_alpha + DIFF_beta * Ee)
    return sigma_xy

@njit
def generate_event(Ee):
    """Generate a randoom position uniform distributed along the x, y and z axis of the CCD
    """
    
    position = np.array([
        np.random.uniform(ROWS[0], ROWS[1]),
        np.random.uniform(COLS[0], COLS[1]),
        np.random.uniform(DEPTH[0],DEPTH[1])
        ])
   	
    sigma_xy = np.sqrt(-DIFF_A * np.log(1-DIFF_b*position[2])) * (DIFF_alpha + DIFF_beta * Ee)
    	
    position_diff = np.array([
        np.random.normal(position[0], sigma_xy),
        np.random.normal(position[1], sigma_xy),
        np.random.normal(position[2], sigma_xy)
        ])
    return np.array([position[0],position[1],position[2],position_diff[0],position_diff[1],position_diff[2]])


@njit(parallel=True)
def generate_multiple_events(Ee):
    """Paralelize generate_event function with njit
    """
    n = len(Ee)
    events = np.empty((n, 6), dtype=np.float64)
    for i in prange(n):
        events[i] = generate_event(Ee[i])

    return events


#################################################################

#Ee = np.repeat(3,1000000)
#_ = generate_multiple_events(Ee)
#_ = generate_multiple_events(Ee)
#_ = [generate_event(i) for i in Ee]
#_ = [generate_event(i) for i in Ee]




#DIFF_A      = 803.3*um*um
#DIFF_b      = 6.5e-4/um
#DIFF_alpha  = 0.8594
#DIFF_beta   = 6.7e-3/eV
