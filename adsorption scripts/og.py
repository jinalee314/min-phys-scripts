#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025
PURPOSE: Extracts and averages default data from the original GDS analysis that utilize coarse-graining along the proximity axis.
Step 3 (after ne.py) in the data post-processing and analysis pipeline.

INPUT:
    - rawdata.npz

OUTPUT:
    - og_fraction.npz (large file, optional, not necessary for the full analysis)
    - og_avg.npz
    - og_frac_total_avg_reprox.png

Note:
The data handled in this script does not preserve any information on individual Ne atoms, so the output generated from ne.py
is not needed here. The density of the system and molar fractions of O, Mg, Fe, and Ne as a function of proximity coordinate
are very smoothed out due to the coarse-graining length set in the GDS analysis (2.5 Angstrom) and lose detailed information
on any trapped Ne atoms or aberrant Ne proximity coordinates. The Ne concentration profile at the interfacial region appears
artificially smooth and wide and thus this data is not used for the final analysis. The density profile is utilized in later 
parts, but not the molar fraction profiles. This script provides an opportunity to visualize what these profiles look like.

While frame.txt contains the absolute bounds of frame processed in the GDS analysis, the actual frames used in the final analysis
can be selected using the --start and --end arguments. Plots avgs.png and element_counts.png (from extract.py) can provide
insight on an appropriate interval, but the chosen frame interval must be held consistent for the rest of the analysis pipeline.
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str, default='./', help="Path to the directories")
parser.add_argument("--start", "-s", type=int, default=0, help="Start index for processing")
parser.add_argument("--end", "-e", type=int, default=100000000, help="End index for processing")

args = parser.parse_args()
path = args.path
start = args.start
end = args.end

# access raw data generated from extract.py
raw = np.load(os.path.join(path, 'rawdata.npz'))
prox = raw['prox']
Ne_frac = raw['frac_Ne']
Fe_frac = raw['frac_Fe']
Mg_frac = raw['frac_Mg']
O_frac = raw['frac_O']
density = raw['density']
chi_filter = raw['chi_filter']

# subset of frames that will be used throughout the rest of the analysis pipeline
final = [i for i in chi_filter if (i > start and i < end)]

lw = raw['lw']
avg_lw = np.mean(lw[final])

# interpolate all data to a common proximity grid
# the grid bounds are determined by the maximum value of the leftmost proximity coordinate and minimum of the rightmost points
left = int(np.ceil(np.max(prox[final][:,0])))
right = int(np.floor(np.nanmin(prox[final][:,-1])))
num = (right-left)*1000 + 1
xnew = np.linspace(left, right, num)
Mg = []
O = []
Fe = []
Ne = []
d = []
for i in final:
    p = prox[i]
    indices = np.where((p >= left) & (p <= right))
    x = p[indices]

    f = interp1d(x, Mg_frac[i][indices], kind='linear', fill_value="extrapolate")
    Mg.append(f(xnew))
    f = interp1d(x, O_frac[i][indices], kind='linear', fill_value="extrapolate")
    O.append(f(xnew))
    f = interp1d(x, Fe_frac[i][indices], kind='linear', fill_value="extrapolate")
    Fe.append(f(xnew))
    f = interp1d(x, Ne_frac[i][indices], kind='linear', fill_value="extrapolate")
    Ne.append(f(xnew))
    f = interp1d(x, density[i][indices], kind='linear', fill_value="extrapolate")
    d.append(f(xnew))

print('interpolated')

#np.savez_compressed(os.path.join(path,'og_fraction.npz'), xnew=xnew, Mg=Mg, O=O, Fe=Fe, Ne=Ne, density=d)
#np.savez_compressed(os.path.join(path,'og_fraction'+str(start)+'.npz'), xnew=xnew, Mg=Mg, O=O, Fe=Fe, Ne=Ne, density=d)

# can now average all profiles since they share the same proximity grid
Mg_arr = np.array(Mg)
O_arr = np.array(O)
Fe_arr = np.array(Fe)
Ne_arr = np.array(Ne)
d_arr = np.array(d)

avg_Mg = np.mean(Mg_arr, axis=0)
avg_O = np.mean(O_arr, axis=0)
avg_Fe = np.mean(Fe_arr, axis=0)
avg_Ne = np.mean(Ne_arr, axis=0)
avg_d = np.mean(d_arr, axis=0)

np.savez_compressed(os.path.join(path,'og_avg.npz'), xnew=xnew, avg_Mg=avg_Mg, avg_O=avg_O, avg_Fe=avg_Fe, avg_Ne=avg_Ne, avg_density=avg_d)
#np.savez_compressed(os.path.join(path,'og_avg'+str(start)+'.npz'), xnew=xnew, avg_Mg=avg_Mg, avg_O=avg_O, avg_Fe=avg_Fe, avg_Ne=avg_Ne, avg_density=avg_d)

# visualizing averaged atomic fraction profiles for each element
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()
axs[0].plot(xnew, avg_O, label='O', color='blue')
axs[1].plot(xnew, avg_Mg, label='Mg', color='green')
axs[2].plot(xnew, avg_Fe, label='Fe', color='orange')
axs[3].plot(xnew, avg_Ne, label='Ne', color='purple')
axs[0].axvline(x=0, color='black', linestyle='--')
axs[0].axhline(y=0, color='black', linestyle='--')
axs[0].axvline(x=-avg_lw/2, color='red', linestyle='--')
axs[0].axvline(x=avg_lw/2, color='red', linestyle='--')
axs[1].axvline(x=0, color='black', linestyle='--')
axs[1].axhline(y=0, color='black', linestyle='--')
axs[1].axvline(x=-avg_lw/2, color='red', linestyle='--')
axs[1].axvline(x=avg_lw/2, color='red', linestyle='--')
axs[2].axvline(x=0, color='black', linestyle='--')
axs[2].axhline(y=0, color='black', linestyle='--')
axs[2].axvline(x=-avg_lw/2, color='red', linestyle='--')
axs[2].axvline(x=avg_lw/2, color='red', linestyle='--')
axs[3].axvline(x=0, color='black', linestyle='--')
axs[3].axhline(y=0, color='black', linestyle='--')
axs[3].axvline(x=-avg_lw/2, color='red', linestyle='--')
axs[3].axvline(x=avg_lw/2, color='red', linestyle='--')
axs[0].set_ylabel('O Atomic Fraction')
axs[1].set_ylabel('Mg Atomic Fraction')
axs[2].set_ylabel('Fe Atomic Fraction')
axs[3].set_ylabel('Ne Atomic Fraction')
axs[0].set_xlabel('Proximity (Angstrom)')
axs[1].set_xlabel('Proximity (Angstrom)')
axs[2].set_xlabel('Proximity (Angstrom)')
axs[3].set_xlabel('Proximity (Angstrom)')
plt.tight_layout()
plt.savefig(os.path.join(path, "og_frac_total_avg_reprox_"+str(start)+".png"))
plt.close()
