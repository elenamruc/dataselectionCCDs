"""
Original code written by Nuria.

The last section (marked as "# ADDED BY ELENA") was added by me to determine the energy boundaries
where the most probable number of electron-hole (e-h) pairs changes.
"""


import numpy as np

# UNITS
um = 1.0
cm = 10000*um
eV = 1.0

MEAN_Eeh_CREATION = 3.74*eV
FANOFACTOR = 0.12

# IMPORT PROBABILITIES 
#import pydme.data
#data = pydme.data.load_data_file("p100K.npz")
data = np.load("p100K.npz")['data'].T
PAIR_CREATION_PROBABILITIES = {
        'E': data[0,:],
        'P': data[1:,:],
        'Npair_bins': np.arange(1,21)
        }
 

# energy resolution of the given file
PCP_RESOLUTION = 0.05


def plot_Pneh(P,E, xlim=None):
    from matplotlib import pyplot as plt
    

    plt.figure()

    for i,(y,Ee) in enumerate(zip(P,E)):
        plt.bar(PAIR_CREATION_PROBABILITIES['Npair_bins'], P[i], label=f"{Ee} eV", alpha=0.5)
    
    plt.yscale('log')
    plt.xlabel("num. pair creation")
    plt.ylabel("pair creation probability per E")
    plt.legend(loc='best')
    plt.show()
    input("press enter ...")
    
    return

################################################################################ CHARGE IONIZATION
def Probability_Neh_E(E,Niter):
    """Number of electron-hole pair that will be created for a given ionization energy

    Probability distribution: 0 P(ne|E)

    E total energy of the event in units of eV

    """
    
    E = np.round(E/PCP_RESOLUTION) * PCP_RESOLUTION
    if E < 1.1*eV:
        E = 1.1

    # Get the probability to ionize the labeled number of charge pairs for a given deposited energy (E)
    index    = np.searchsorted(PAIR_CREATION_PROBABILITIES['E'], E)
    p_pair_E = PAIR_CREATION_PROBABILITIES['P'][:,index]
    Norm = np.sum(p_pair_E)
    Npair_bins    = PAIR_CREATION_PROBABILITIES['Npair_bins']
    
    rng = np.random.default_rng()
    
    if Norm>0.0:
        p_pair_E_norm = p_pair_E / Norm
        neh = rng.choice(Npair_bins, size=Niter, p=p_pair_E_norm)
        # frequency for the 0 case is also included
        fne = np.bincount(neh,minlength=len(Npair_bins)+1)/Niter
        fne = fne[1:]
    else:
        fne = np.zeros_like(Npair_bins)

    return fne

def build_pair_creation_probabilities(Energy,Niter,outfile=None,plot=False):
    """Function to build the probability functions to create the number of pair for a given energy lost (Energy)

    Energy in units of eV. 
        The resolution should not be lower than the resolution of the probability functions, i.e. 0.05eV 
    Niter the number of random choices to estimage the frequency for each number of e-h created pairs (1,2,3,...,20)
    
    outfile is the npz output file where these probabilities will be stored as a dictionary {E: P(ne|E) }

    plot set to display the estimated frequencies
    
    """
    
    P_Neh = []
    if Energy is None:
        Energy = np.round(np.arange(1.10,10.00,0.05),2)
    
    for i,E in enumerate(Energy):
        P_Neh.append( Probability_Neh_E(E,Niter) )

    if outfile:
        data = {f"{Energy[i]:.2f}": P_Neh[i] for i in range(len(Energy))}
        data.update({'Ebins': [f"{Energy[i]:.2f}" for i in range(len(Energy))] })
        np.savez(outfile, **data)
    
    if plot:
        _ = plot_Pneh(P_Neh,Energy)

    return Energy,P_Neh
    

################################################################################
# ADDED BY ELENA
def find_energy_ranges_for_pairs(E_min=1.1, E_max=20.0, step=0.05, Niter=50000):
    """
    Scans an energy range and determines the most likely number of e-h pairs
    created at each energy. Returns approximate energy thresholds where the
    mode of the distribution transitions from 1 to 2, 2 to 3, and so on.

    Returns:
    - Elist: list of scanned energy values
    - most_likely_pairs: array of most probable pair counts
    - transitions: list of (n_old, n_new, E_transition) tuples
    """
    Elist = np.arange(E_min, E_max + step, step)
    most_likely_pairs = []

    for E in Elist:
        p = Probability_Neh_E(E, Niter)
        n = np.argmax(p)  # index 0 = 1 pair, so add +1
        most_likely_pairs.append(n)

    most_likely_pairs = np.array(most_likely_pairs)

    transitions = []
    last_n = most_likely_pairs[0]
    for i in range(1, len(Elist)):
        n_current = most_likely_pairs[i]
        if n_current != last_n:
            E_boundary = 0.5*(Elist[i-1] + Elist[i])
            transitions.append((last_n, n_current, E_boundary))
            last_n = n_current

    return Elist, most_likely_pairs, transitions
    
    
if __name__ == "__main__":

    Elist, mlp, transitions = find_energy_ranges_for_pairs(
        E_min=1.1, E_max=25.0, step=0.05, Niter=50000
    )

    print("Transitions (from n to n+1 pairs) found:")
    for (n_old, n_new, E_boundary) in transitions:
        print(f"   {n_old} -> {n_new} a E ~ {E_boundary:.3f} eV")
