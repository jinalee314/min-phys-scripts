#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025
PURPOSE: Compute surface excess and Ne partitioning from simulation data
Step 6 (final) in the data post-processing and analysis pipeline.

INPUT:
    - rawdata.npz (from extract.py)
    - avg.npz (from frac.py)
    - density_fit.npy (from density_fit.py)

OUTPUT:
    - avg_region_Mg.png
    - avg_region_O.png
    - avg_region_Fe.png
    - avg_region_Ne.png
    - terminal output (copied into final.txt)
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str, default='./', help="Path to the directories")
parser.add_argument("--start", "-s", type=int, default=0, help="Start index for processing")
parser.add_argument("--end", "-e", type=int, default=100000000, help="End index for processing")
parser.add_argument("--trapped", "-t", type=int, default=0, help="number of trapped Ne atoms")

args = parser.parse_args()
path = args.path
start = args.start
end = args.end
trapped = args.trapped

# access raw data generated from extract.py
raw = np.load(os.path.join(path, 'rawdata.npz'))
Mg = raw['Mg']
O = raw['O']
Fe = raw['Fe']
Ne = raw['Ne']
lw = raw['lw']
chi_filter = raw['chi_filter']

# subset of frames used throughout the analysis pipeline
final = [i for i in chi_filter if (i > start and i < end)]

# automatically loads modified Ne proximity data
if os.path.isfile(os.path.join(path, 'ne.npz')):
    ne_mod = np.load(os.path.join(path, 'ne.npz'))
    Ne = ne_mod['Ne']
    print("Using modified Ne phase data from ne.npz")


# averaging atom counts per phase: format [liquid, solid, interface] per frame
Mg_l = np.mean(Mg[final,0])
Mg_s = np.mean(Mg[final,1])
O_l = np.mean(O[final,0])
O_s = np.mean(O[final,1])
Fe_l = np.mean(Fe[final,0])
Fe_s = np.mean(Fe[final,1])
Ne_l = np.mean(Ne[final,0])
Ne_s = np.mean(Ne[final,1])
Ne_int = np.mean(Ne[final,2])
total_l = Mg_l + O_l + Fe_l + Ne_l
total_s = Mg_s + O_s + Fe_s + Ne_s

# mole fractions per phase
x_Mg_l = Mg_l/total_l
x_Mg_s = Mg_s/total_s
x_O_l = O_l/total_l
x_O_s = O_s/total_s
x_Fe_l = Fe_l/total_l
x_Fe_s = Fe_s/total_s
x_Ne_l = Ne_l/total_l # Ne in bulk Fe
x_Ne_s = Ne_s/total_s # Ne in bulk ferropericlase
x_Ne_int = Ne_int/total_s # interfacial Ne computed with respect to bulk ferropericlase

# output results computed based on counts allocated to each phase
print('COUNT BASED')
print('solid molfrac:',x_Mg_s, x_O_s, x_Fe_s, x_Ne_s, x_Ne_int)
print('liquid molfrac:',x_Mg_l, x_O_l, x_Fe_l, x_Ne_l)

# accounting for trapped Ne in ferropericlase
x_Ne_s_bulk = x_Ne_s
if trapped > 0:
    x_Ne_s = (Ne_s-trapped)/total_s
    print('recomputed solid molfrac (omit trapped):',x_Ne_s)


# recomputed molar fraction profiles wrt proximity, averaged in frac.py
avg = np.load(os.path.join(path, 'avg.npz'))
avg_prox = avg['xnew']
avg_Mg = avg['avg_Mg']
avg_O = avg['avg_O']
avg_Fe = avg['avg_Fe']
avg_Ne = avg['avg_Ne']
avg_Ne_t = avg['avg_Ne_t']

# average mole fractions in the bulk liquid based on proximity range
# for comparison with count-based results
region_r = avg_prox > np.mean(lw[final])/2 # approximate threshold for averaging
mean_Mg = np.mean(avg_Mg[region_r])
mean_O = np.mean(avg_O[region_r])
mean_Fe = np.mean(avg_Fe[region_r])
mean_Ne = np.mean(avg_Ne[region_r])
print("PROFILE BASED")
print("liquid mean:", mean_Mg, mean_O, mean_Fe, mean_Ne)

# plotting average profiles for the two bulk regions
# to get a sense of what they look like, how smooth they are, etc.
region1 = avg_prox < -2
region2 = avg_prox > 2

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(avg_prox[region1], avg_Mg[region1], label='Mg', color='green')
axs[1].plot(avg_prox[region2], avg_Mg[region2], label='Mg', color='green')
axs[1].axhline(y=mean_Mg, color='red', linestyle='--')
plt.xlabel('Proximity (Angstrom)')
axs[0].set_ylabel('Mg Atomic Fraction')
axs[1].set_ylabel('Mg Atomic Fraction')
plt.tight_layout()
plt.savefig(os.path.join(path, "avg_region_Mg.png"))
plt.close()

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(avg_prox[region1], avg_O[region1], label='O', color='blue')
axs[1].plot(avg_prox[region2], avg_O[region2], label='O', color='blue')
axs[1].axhline(y=mean_O, color='red', linestyle='--')
plt.xlabel('Proximity (Angstrom)')
axs[0].set_ylabel('O Atomic Fraction')
axs[1].set_ylabel('O Atomic Fraction')
plt.tight_layout()
plt.savefig(os.path.join(path, "avg_region_O.png"))
plt.close()

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(avg_prox[region1], avg_Fe[region1], label='Fe', color='orange')
axs[1].plot(avg_prox[region2], avg_Fe[region2], label='Fe', color='orange')
plt.xlabel('Proximity (Angstrom)')
axs[0].set_ylabel('Fe Atomic Fraction')
axs[1].set_ylabel('Fe Atomic Fraction')
axs[1].axhline(y=mean_Fe, color='red', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(path, "avg_region_Fe.png"))
plt.close()

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(avg_prox[region1], avg_Ne[region1], label='Ne', color='purple')
axs[1].plot(avg_prox[region2], avg_Ne[region2], label='Ne', color='purple')
axs[1].axhline(y=mean_Ne, color='red', linestyle='--')
plt.xlabel('Proximity (Angstrom)')
axs[0].set_ylabel('Ne Atomic Fraction')
axs[1].set_ylabel('Ne Atomic Fraction')
plt.tight_layout()
plt.savefig(os.path.join(path, "avg_region_Ne.png"))
plt.close()


# newly fitted density profile
density = np.load(os.path.join(path, 'density_fit.npy'))
ind = np.argmin(np.abs(avg_prox)) # index where proximity = 0

# bulk Ne molar fractions in the solid and liquid phases are used to identify integration bounds
threshold_left = x_Ne_s
threshold_right = x_Ne_l
print('\n')
print('threshold s, l used:', threshold_left, threshold_right)
left_i = 0
middle_i = ind 
right_i = 0
for i in range(ind, 0, -1):
    if avg_Ne[i] < threshold_left:
        left_i = i
        break
for i in range(ind, len(avg_prox)):
    if avg_Ne[i] < threshold_right:
        right_i = i
        break
print("integration bounds:",avg_prox[left_i], avg_prox[right_i]) # proximity coordinates for integration bounds

# gathering data for integration at the left of the interface and right of the interface
x1, Mg1, O1, Fe1, Ne1, density1 = avg_prox[left_i:middle_i], avg_Mg[left_i:middle_i], avg_O[left_i:middle_i], avg_Fe[left_i:middle_i], avg_Ne[left_i:middle_i], density[left_i:middle_i]
x2, Mg2, O2, Fe2, Ne2, density2 = avg_prox[middle_i:right_i], avg_Mg[middle_i:right_i], avg_O[middle_i:right_i], avg_Fe[middle_i:right_i], avg_Ne[middle_i:right_i], density[middle_i:right_i]

# integration to compute surface excess at left side of interface 
integrand1 = density1 * 1e-24 * (Ne1 - threshold_left) / (Mg1 * 24.305 + O1 * 15.999 + Fe1 * 55.845 + Ne1 * 20.1797)
n_ex1 = trapezoid(integrand1, x1) * 1e20

# integration to compute surface excess at right side of interface 
integrand2 = density2 * 1e-24 * (Ne2 - threshold_right) / (Mg2 * 24.305 + O2 * 15.999 + Fe2 * 55.845 + Ne2 * 20.1797)
n_ex2 = trapezoid(integrand2, x2) * 1e20

# total surface excess
n_ex = n_ex1 + n_ex2

print('\n')
print("left, right excess:",n_ex1, n_ex2)
print("total surface excess:",n_ex)

# various partition coefficient calculations

# basic bulk partition coefficient
bulk_D = x_Ne_l/(x_Ne_s)
print("bulk:",bulk_D)
bulk_D_trapped = 0
if trapped > 0:
    bulk_D_trapped = x_Ne_l/(x_Ne_s_bulk)
    print("bulk trapped:",bulk_D_trapped)

# effective "bulk" partition coefficient including interfacial Ne
eff = x_Ne_l/(x_Ne_s+x_Ne_int)
print("estimate effective:",eff)
if trapped > 0:
    eff_trapped = x_Ne_l/(x_Ne_s_bulk+x_Ne_int)
    print("trapped effective:",eff_trapped)

# Ne concentration in Fe phase
C_Ne_Fe_m = Ne_l/(Mg_l*24.305 + O_l*15.999 + Fe_l*55.845 + Ne_l*20.1797) # mol/g
C_Ne_Fe = C_Ne_Fe_m * density[-1] * 1e3

# geometry informed partition coefficient (utilizes specific surface area) for interfacial Ne only
r = 2e-9 # units: m, approximate size of MgO nanoparticle
mass_Fe_over_MgO = 55.845 / 40.3044
rho_MgO_over_Fe_core = 5.4 / 11.3
D_int = r / 3 * mass_Fe_over_MgO * rho_MgO_over_Fe_core * (C_Ne_Fe * 1e3 / n_ex)

print('\n')
print('Surface excess adsorption:', n_ex  * 1e6, 'μmol m^-2')
print('Ne concentration in Fe:', C_Ne_Fe, 'mol/L;', C_Ne_Fe_m, 'mol/g')
print('Partition coefficient (Earth):', D_int)

# effective partition coefficient including bulk partitioning and geometry informed interfacial Ne contribution
D_eff = D_int * bulk_D/(D_int + bulk_D)
print('Effective partition coefficient FINAL:', D_eff)
if trapped > 0:
    D_eff = D_int * bulk_D_trapped/(D_int + bulk_D_trapped)
    print('Effective partition coefficient FINAL (trapped):', D_eff)
