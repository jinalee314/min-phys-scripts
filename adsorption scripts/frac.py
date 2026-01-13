#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025
PURPOSE: Recomputes element counts and molar fractions from raw proximity coordinates for each atom in the system.
Step 4 (after og.py) in the data post-processing and analysis pipeline.

INPUT:
    - rawdata.npz
    
OUTPUT:
    - counts.npz (optional)
    - fraction.npz (large file, optional)
    - avg.npz
    - frac_total_avg_reprox.png
    - fraction_trapped.npz (optional, if applicable)
    - counts_trapped.npz (optional, if applicable)
    - frac_Ne_trapped_avg.png (if applicable)

Note:
Actual frames used in the final analysis can be selected using the --start and --end arguments. Plots avgs.png and 
element_counts.png (from extract.py) can provide insight on an appropriate interval, but the chosen frame interval 
must be held consistent for the rest of the analysis pipeline.

To account for trapped Ne atoms, use --trapped or -t 1 flag.
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
parser.add_argument("--trapped", "-t", type=int, default=0, help="accounting for trapped Ne")

args = parser.parse_args()
path = args.path
start = args.start
end = args.end
trapped = args.trapped

# access raw data generated from extract.py
raw = np.load(os.path.join(path, 'rawdata.npz'))
atom_prox = raw['atom_prox'] # all information on individual atom proximity coordinates contained here
atom_type = raw['atom_type']
prox = raw['prox']
density = raw['density']
chi_filter = raw['chi_filter']

# automatically loads modified Ne proximity data
if os.path.isfile(os.path.join(path, 'ne.npz')):
    ne_mod = np.load(os.path.join(path, 'ne.npz'))
    atom_prox = ne_mod['atom_prox']
    print("Using modified Ne proximity data from ne.npz")

# loads modified atom types if trapped Ne analysis is desired
if trapped:
    type_switch = np.load(os.path.join(path, 'atom_type_mod.npy'))
    atom_type = type_switch
    print("Using modified atom types from atom_type_mod.npy")

# subset of frames that will be used throughout the rest of the analysis pipeline
final = [i for i in chi_filter if (i > start and i < end)]
#print(len(final))
lw = raw['lw']
avg_lw = np.mean(lw[final])

# recomputing element counts along the proximity axis from scratch (directly from atom_prox)
Mg_i = np.where(atom_type==0)[0]
O_i = np.where(atom_type==1)[0]
Fe_i = np.where(atom_type==2)[0]
Ne_i = np.where(atom_type==3)[0]
Ne_t = np.where(atom_type==4)[0] # remains zero if -t flag is not invoked
count_Ne = []
count_Fe = []
count_O = []
count_Mg = []
count_Ne_t = []
new_prox = []
new_density = []

# creates higher resolution proximity grids and re-bins densities and element counts
# the new "coarse-graining length" is tied to the resolution of this new proximity grid
for i in range(len(prox)):
    # new proximity grid will have resolution 2 times finer than original (q_interval*2)
    q_interval = (prox[i][1] - prox[i][0])/4 # original interval divided by 4

    #if i==3078: # handling of some anomalous frame
     #  q_interval = 0.802842-0.1049326

    original_coords = prox[i]
    # check for NaNs in prox[i]
    if np.isnan(prox[i]).any():
        last_valid_idx = np.where(~np.isnan(prox[i]))[0][-1] # last consecutive non NaN value
        calibrate = 0
        for frame in range(last_valid_idx, 0, -1):
            # some proximity values on the grid are missing and thus not evenly spaced, might be main contributor to NaNs
            if prox[i][frame]-prox[i][frame-1] > q_interval*4: 
                calibrate = frame-1 # index of lowest proximity coordinate preceding any skip in the proximity grid
        for frame in range(calibrate+1, len(prox[i])):
            prox[i][frame] = prox[i][frame-1]+q_interval*4 # readjusts proximity grid to be at even intervals (original resolution)

    # creating higher resolution proximity grid values (bin_centers) and the edges for the histogram count (bin_edges)
    bin_edges = np.concatenate(([prox[i][0] - q_interval], np.arange(prox[i][0]+q_interval, prox[i][-1]+q_interval+1e-8, 2*q_interval)))
    bin_centers = np.arange(prox[i][0], prox[i][-1]+q_interval+1e-8, 2*q_interval)
    new_prox.append(bin_centers)

    # rescaling density onto new proximity grid using linear interpolation
    if np.isnan(original_coords).any():
        # handling NaNs in density and proximity coordinates for proper interpolation
        valid_mask = ~np.isnan(density[i]) & ~np.isnan(original_coords)
        interp_func = interp1d(original_coords[valid_mask], density[i][valid_mask], kind='linear', fill_value="extrapolate")
        density_interp = interp_func(bin_centers)
        new_density.append(density_interp)
        print('recalibrated prox and density',i)
    else:
        interp_func = interp1d(original_coords, density[i], kind='linear', fill_value="extrapolate")
        density_interp = interp_func(bin_centers)
        new_density.append(density_interp)

    Mg_prox = atom_prox[i][Mg_i]
    O_prox = atom_prox[i][O_i]
    Fe_prox = atom_prox[i][Fe_i]
    Ne_prox = atom_prox[i][Ne_i]
    Ne_trap = atom_prox[i][Ne_t] # remains zero if -t flag is not invoked

    # manually rebinning atom count based on their locations along the higher resolution proximity grid (set by bin_edges)
    Mg, _ = np.histogram(Mg_prox, bins=bin_edges)
    O, _ = np.histogram(O_prox, bins=bin_edges)
    Fe, _ = np.histogram(Fe_prox, bins=bin_edges)
    Ne, _ = np.histogram(Ne_prox, bins=bin_edges)
    Ne_trapped, _ = np.histogram(Ne_trap, bins=bin_edges) # remains zero if -t flag is not invoked

    count_Mg.append(Mg)
    count_O.append(O)
    count_Fe.append(Fe)
    count_Ne.append(Ne)
    count_Ne_t.append(Ne_trapped) # remains zero if -t flag is not invoked


Ne_total = np.array(count_Ne) + np.array(count_Ne_t)
total = Ne_total + np.array(count_Fe) + np.array(count_O) + np.array(count_Mg)

#np.savez_compressed(os.path.join(path,'counts.npz'), new_prox=new_prox, count_Mg=count_Mg, count_O=count_O, count_Fe=count_Fe, count_Ne=Ne_total, density=new_density)
#np.savez_compressed(os.path.join(path,'counts'+str(start)+'.npz'), new_prox=new_prox, count_Mg=count_Mg, count_O=count_O, count_Fe=count_Fe, count_Ne=Ne_total, density=new_density)
#if trapped:
#    np.savez_compressed(os.path.join(path,'counts_trapped.npz'), count_Ne=count_Ne, count_Ne_t=count_Ne_t)


# converting atom counts into atomic fractions
Ne_frac = Ne_total/total
Fe_frac = np.array(count_Fe)/total
Mg_frac = np.array(count_Mg)/total
O_frac = np.array(count_O)/total

Ne_frac_t = []
Ne_frac_omit = []
if trapped:
    Ne_frac_t = np.array(count_Ne_t)/total # atomic fraction profile of just the trapped Ne atoms alone
    Ne_frac_omit = np.array(count_Ne)/total # atomic fraction profile of Ne ignoring the trapped Ne atoms

# interpolate all data to a common proximity grid
# the grid bounds are determined by the maximum value of the leftmost proximity coordinate and minimum of the rightmost points
left = int(np.ceil(np.max(new_prox[final][:,0])))
right = int(np.floor(np.nanmin(new_prox[final][:,-1])))
num = (right-left)*1000 + 1
xnew = np.linspace(left, right, num)
Mg = []
O = []
Fe = []
Ne = []
Ne_t = []
Ne_omit = []
d = []

for i in final:
    p = new_prox[i]
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
    f = interp1d(x, new_density[i][indices], kind='linear', fill_value="extrapolate")
    d.append(f(xnew))
    if trapped:
        f = interp1d(x, Ne_frac_t[i][indices], kind='linear', fill_value="extrapolate")
        Ne_t.append(f(xnew))
        f = interp1d(x, Ne_frac_omit[i][indices], kind='linear', fill_value="extrapolate")
        Ne_omit.append(f(xnew))

print('interpolated')

#np.savez_compressed(os.path.join(path,'fraction.npz'), xnew=xnew, Mg=Mg, O=O, Fe=Fe, Ne=Ne, density=d)
#np.savez_compressed(os.path.join(path,'fraction'+str(start)+'.npz'), xnew=xnew, Mg=Mg, O=O, Fe=Fe, Ne=Ne, density=d)
#if trapped:
#    np.savez_compressed(os.path.join(path,'fraction_trapped.npz'), Ne_t=Ne_t, Ne_omit=Ne_omit)


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

avg_Ne_t = np.zeros(num)
avg_Ne_omit = np.zeros(num)

if trapped:
    Ne_t_arr = np.array(Ne_t)
    Ne_omit_arr = np.array(Ne_omit)
    avg_Ne_t = np.mean(Ne_t_arr, axis=0)
    avg_Ne_omit = np.mean(Ne_omit_arr, axis=0)


# includes 3 types of Ne profiles:
# Ne profile with all Ne atoms, Ne profile with only trapped Ne atoms, and Ne profile omitting trapped Ne atoms
np.savez_compressed(os.path.join(path,'avg.npz'), xnew=xnew, avg_Mg=avg_Mg, avg_O=avg_O, avg_Fe=avg_Fe, avg_Ne=avg_Ne, avg_Ne_t=avg_Ne_t, avg_Ne_omit=avg_Ne_omit, avg_density=avg_d)
#np.savez_compressed(os.path.join(path,'avg'+str(start)+'.npz'), xnew=xnew, avg_Mg=avg_Mg, avg_O=avg_O, avg_Fe=avg_Fe, avg_Ne=avg_Ne, avg_Ne_t=avg_Ne_t, avg_Ne_omit=avg_Ne_omit, avg_density=avg_d)


# visualizing averaged atomic fraction profiles for each element, rebinned from original raw data
if trapped:
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(xnew, avg_Ne_omit, label='Ne (omit trapped)', color='purple')
    axs[0].set_ylabel('Ne Atomic Fraction (omit trapped)')
    axs[0].axvline(x=0, color='black', linestyle='--')
    axs[0].axhline(y=0, color='black', linestyle='--')
    axs[0].axvline(x=-avg_lw/2, color='red', linestyle='--')
    axs[0].axvline(x=avg_lw/2, color='red', linestyle='--')
    axs[0].legend()

    axs[1].plot(xnew, avg_Ne_t, label='Ne (trapped)', color='orange')
    axs[1].set_ylabel('Ne Atomic Fraction (trapped)')
    axs[1].set_xlabel('Proximity (Angstrom)')
    axs[1].axvline(x=0, color='black', linestyle='--')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].axvline(x=-avg_lw/2, color='red', linestyle='--')
    axs[1].axvline(x=avg_lw/2, color='red', linestyle='--')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, "frac_Ne_trapped_avg.png"))
    plt.close()
    

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
plt.savefig(os.path.join(path, "frac_total_avg_reprox_"+str(start)+".png"))
plt.close()
