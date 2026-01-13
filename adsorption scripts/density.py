#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025
PURPOSE: Refits density profile obtained from coarse-grained GDS analysis to better align with recomputed molar fractions
from frac.py
Step 5 (after frac.py) in the data post-processing and analysis pipeline.

INPUT:
    - rawdata.npz
    - avg.npz
    - og_avg.npz

OUTPUT:
    - avg_mass_fit.png
    - avg_refit_Fe_density.png
    - bulk_summary.png
    - bulk_density_fe_fit.png
    - bulk_density_fp_fit.png
    - density_fit.npy
    - final_density_fit.png

Note:
Instead of directly trying to recompute absolute density from the raw data, this script takes a simpler and more 
approximate approach of refitting the density profile originally obtained from GDS analysis. The original density
profile is more "smoothed" over and lacks the sharper transitions seen in the recomputed molar fractions from frac.py.
That density profile is compressed along the proximity axis (by a factor "a" set in the code) to better mimic the 
recomputed interfacial transition. This refitted profile will be utilized in the next step of the analysis pipeline.

This compression will shrink the range of x values covered by the density profile, so extrapolations will be needed
to cover the rest of the original proximity range into the bulk regions. This is where the tags come in.
- mean_left (ml) and mean_right (mr) set the windows over which the bulk density is averaged to get asymptotic plateau values
- x_left (xl) and x_right (xr) set the anchor points on the density curve from which the extrapolations begin
- inf_left (il) and inf_right (ir) set the inflection points of the logistic fit
- k_left (kl) and k_right (kr) set the steepness of the logistic curves
The current default values set for these tags work for the system located at 
/scratch/gpfs/JIEDENG/jina/adsorption/simulations/8000_atom/10ne
Visual inspection of the plots is necessary in order to find appropriate values for these tags that leads to a good fit.
This could be a more automated process in the future, but for now it is manual.
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str, default='./', help="Path to the directories")
parser.add_argument("--start", "-s", type=int, default=0, help="Start index for processing")
parser.add_argument("--end", "-e", type=int, default=100000000, help="End index for processing")
parser.add_argument("--mean_left","-ml", type=float, default=-2.2, help="left coordinate")
parser.add_argument("--mean_right","-mr", type=float, default=2.0, help="right coordinate")
parser.add_argument("--x_left", "-xl", type=float, default=-2.5, help="left anchor point")
parser.add_argument("--x_right", "-xr", type=float, default=2.3, help="right anchor point")
parser.add_argument("--inf_left", "-il", type=float, default=-3.0, help="left inflection point")
parser.add_argument("--inf_right", "-ir", type=float, default=2.6, help="right inflection point")
parser.add_argument("--k_left", "-kl", type=float, default=5, help="steepness of left logistic fit")
parser.add_argument("--k_right", "-kr", type=float, default=8, help="steepness of right logistic fit")

args = parser.parse_args()
start = args.start
end = args.end
path = args.path

# access raw data generated from extract.py
raw = np.load(os.path.join(path, 'rawdata.npz'))
chi_filter = raw['chi_filter']
lw = raw['lw']

# subset of frames that will be used throughout the rest of the analysis pipeline
final = [i for i in chi_filter if (i > start and i < end)]
avg_lw = np.mean(lw[final])

# recomputed profiles wrt to proximity, averaged in frac.py
avg = np.load(os.path.join(path, 'avg.npz'))
prox = avg['xnew']
Mg = avg['avg_Mg']
O = avg['avg_O']
Fe = avg['avg_Fe']
Ne = avg['avg_Ne']

# original proximity profile from GDS analysis, averaged in og.py
og_avg = np.load(os.path.join(path, 'og_avg.npz'))
og_prox = og_avg['xnew']
og_Mg = og_avg['avg_Mg']
og_O = og_avg['avg_O']
og_Fe = og_avg['avg_Fe']
og_Ne = og_avg['avg_Ne']
density = og_avg['avg_density']

# system molar mass along the proximity coordinate
mass = 24.305*Mg + 15.999*O + 55.845*Fe + 20.1797*Ne
og_mass = 24.305*og_Mg + 15.999*og_O + 55.845*og_Fe + 20.1797*og_Ne

# approximate refit by compressing profile along the proximity axis
a = 0.33 # adjust compression factor as needed, with visual inspection of the plots below
x_compressed = a * og_prox

# comparison of avg molar mass profiles
# mass is directly proportional to density, so aligning the blue and green profiles as 
# closely as possible will determine the fit of the new density profile
plt.plot(prox, mass, label='new avg')
plt.plot(og_prox, og_mass, label='og avg')
plt.plot(x_compressed, og_mass, label='Compressed og_prox*'+str(a))
plt.legend()
plt.xlabel('Proximity (Angstrom)')
plt.ylabel('Average molar mass (g/mol)')
plt.axvline(x=0, color='black', linestyle='--')
plt.axvline(x=avg_lw/2, color='red', linestyle='--')
plt.axvline(x=-avg_lw/2, color='red', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(path, "avg_mass_fit.png"))
plt.close()

# comparison of avg Fe molar fraction profile and refitted density profile
# Fe as a single element is a decent proxy for how the density varies across the interface
shift = density-np.min(density)
plt.plot(prox,  Fe/np.max(Fe), label='avg Fe molar fraction')
plt.plot(x_compressed, shift/np.max(shift), label='refitted density')
plt.axvline(x=0, color='black', linestyle='--')
plt.axvline(x=avg_lw/2, color='red', linestyle='--')
plt.axvline(x=-avg_lw/2, color='red', linestyle='--')
plt.legend()
plt.xlabel('Proximity (Angstrom)')
plt.ylabel('Normalized')
plt.tight_layout()
plt.savefig(os.path.join(path, "avg_refit_Fe_density.png"))
plt.close()


# plots showing the density transition between interfacial and bulk regions along the proximity axis
# shown for both Fp side of the interface (left) and Fe side of the interface (right)
# bottom 2 subplots zoom up to the bulk regions only, where the bounds of the regions visualized are set by
# args.mean_left and args.mean_right for Fp and Fe respectively
# the purpose of these plots is to identify, via visual inspection, reasonable points on the curve to anchor
# the logistic fits later on (picking args.x_left and args.x_right)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# fp interface-bulk region
x_fp = x_compressed < -1.0
density_fp = density[x_fp]
mean_fp = np.mean(density_fp) 
axs[0, 0].scatter(x_compressed[x_fp], density_fp,s=1, alpha=0.5)
axs[0, 0].axhline(y=mean_fp, color='red', linestyle='--') # for visual guiding purposes
axs[0, 0].set_xlabel('Transition region (Angstrom)')
axs[0, 0].set_ylabel('Fp Density (g/cm$^3$)')

# Fe interface-bulk region
x_fe = x_compressed > 1.0
density_fe = density[x_fe]
mean_fe = np.mean(density_fe)
axs[0, 1].scatter(x_compressed[x_fe], density_fe,s=1, alpha=0.5)
axs[0, 1].axhline(y=mean_fe, color='red', linestyle='--')
axs[0, 1].set_xlabel('Transition region (Angstrom)')
axs[0, 1].set_ylabel('Fe Density (g/cm$^3$)')

# fp zoomed bulk region
x_fp = x_compressed < args.mean_left
density_fp = density[x_fp]
mean_fp = np.mean(density_fp) # mean value used for the asymptotic plateau of the logistic fit
axs[1, 0].scatter(x_compressed[x_fp], density_fp,s=1, alpha=0.5)
axs[1, 0].axhline(y=mean_fp, color='red', linestyle='--')
axs[1, 0].set_xlabel('Bulk region (Angstrom)')
axs[1, 0].set_ylabel('Fp Density (g/cm$^3$)')

# Fe zoomed bulk region
x_fe = x_compressed > args.mean_right
density_fe = density[x_fe]
mean_fe = np.mean(density_fe)  # mean value used for the asymptotic plateau of the logistic fit
axs[1, 1].scatter(x_compressed[x_fe], density_fe,s=1, alpha=0.5)
axs[1, 1].axhline(y=mean_fe, color='red', linestyle='--')
axs[1, 1].set_xlabel('Bulk region (Angstrom)')
axs[1, 1].set_ylabel('Fe Density (g/cm$^3$)')

plt.tight_layout()
plt.savefig(os.path.join(path, "bulk_summary.png"))
plt.close()


# Logistic-like functional fits to extrapolate the density profile smoothly into the bulk regions
# anchored at points args.x_left and args.x_right on the curve, for the fp and Fe sides respectively
# inflection points set at args.inf_left and args.inf_right, for the fp and Fe sides respectively
# steepness of the logistic curve extrapolations set by args.k_left and args.k_right
# all of these tags need to be toggled using visual inspection to get a smooth fit
def logistic_plateau_anchored(x, x0, y0, y_inf, x_mid, k, direction='right'):
    """
    starts exactly at (x0, y0) and approaches y_inf asymptotically
    """
    # flips direction if plateau is toward left
    sign = 1 if direction == 'right' else -1
    
    # base logistic
    L = 1 / (1 + np.exp(-sign * k * (x - x_mid)))
    L_start = 1 / (1 + np.exp(-sign * k * (x0 - x_mid)))
    
    # anchored form
    y = y0 + (y_inf - y0) * (L - L_start) / (1 - L_start)
    return y


# right side extrapolation (Fe side)
region = x_compressed > args.x_right
x = x_compressed[region]
y = density[region]

y0 = y[0]       # starting y value for anchor point at the left
x_mid = args.inf_right # x value at inflection point
k = args.k_right # steepness of extrapolated curve
ind_max = prox[-1] # end of the original proximity range (refitted range is compressed to a smaller x range)

x_ext_fe = np.linspace(x[0], ind_max, 800) # fine grid for smooth extrapolation
y_ext_fe = logistic_plateau_anchored(x_ext_fe, x[0], y0, mean_fe, x_mid, k)

# subplots to help with visual inspection of the logistic fit on the Fe side
fig, axs = plt.subplots(2, 1, figsize=(6, 8))
axs[0].plot(x_ext_fe[:100], y_ext_fe[:100], label='fit') # zoomed in view near anchor point
axs[0].axvline(x=x_mid, color='green', linestyle='--', label='inflection')
axs[0].axhline(y=mean_fe, color='red', linestyle='--', label='bulk density', alpha=0.5)
axs[0].scatter(x, y, label='data', s=1, alpha=0.5,color='orange') # raw data as shown in bulk_summary.png
axs[0].set_ylabel('Fe Density (g/cm$^3$)')
axs[0].legend()
# combined view of extrapolated fit connected to the original data in transition region
# want the connection to be as smooth as possible
region = (x_compressed <= args.x_right) & (x_compressed > 2.0) # transition region with raw data
combined_x = np.concatenate((x_compressed[region], x_ext_fe[:200]))
combined_y = np.concatenate((density[region], y_ext_fe[:200]))
axs[1].plot(combined_x, combined_y, label='combined')
axs[1].scatter(x, y, label='data', s=1, alpha=0.5, color='orange')
axs[1].axhline(y=mean_fe, color='red', linestyle='--', label='bulk density', alpha=0.5)
axs[1].set_xlabel('Proximity (Angstrom)')
axs[1].set_ylabel('Fe Density (g/cm$^3$)')
axs[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(path, "bulk_density_fe_fit.png"))
plt.close()

# left side extrapolation (Fp side)
region = x_compressed < args.x_left
x = x_compressed[region]
y = density[region]

y0 = y[-1]       # starting y value for anchor point at the right
x_mid = args.inf_left
k = args.k_left
ind_min = prox[0] # x value at far left end of the original proximity range

x_ext_fp = np.linspace(ind_min, x[-1], 800) 
y_ext_fp = logistic_plateau_anchored(x_ext_fp, x[-1], y0, mean_fp, x_mid, k, direction='left')

# subplots to help with visual inspection of the logistic fit on the fp side
fig, axs = plt.subplots(2, 1, figsize=(6, 8))
axs[0].plot(x_ext_fp[400:], y_ext_fp[400:], label='fit')
axs[0].axvline(x=x_mid, color='green', linestyle='--', label='inflection')
axs[0].axhline(y=mean_fp, color='red', linestyle='--', label='bulk density', alpha=0.5)
axs[0].scatter(x, y, label='data', s=1, alpha=0.5,color='orange')
axs[0].set_ylabel('Fp Density (g/cm$^3$)')
axs[0].legend()
# combined view of extrapolated fit connected to the original data in transition region
region = (x_compressed >= args.x_left) & (x_compressed < -2.0) # transition region with raw data
combined_x = np.concatenate((x_ext_fp[500:],x_compressed[region]))
combined_y = np.concatenate((y_ext_fp[500:],density[region]))
axs[1].plot(combined_x, combined_y, label='combined')
axs[1].scatter(x, y, label='data', s=1, alpha=0.5, color='orange')
axs[1].axhline(y=mean_fp, color='red', linestyle='--', label='bulk density', alpha=0.5)
axs[1].set_ylabel('Fp Density (g/cm$^3$)')
axs[1].set_xlabel('Proximity (Angstrom)')
axs[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(path, "bulk_density_fp_fit.png"))
plt.close()


# combine the three pieces to the refitted density profile
middle = (x_compressed >= args.x_left) & (x_compressed <= args.x_right) # transition region
middle_x = x_compressed[middle]
middle_y = density[middle]

final_x = np.concatenate((x_ext_fp, middle_x, x_ext_fe))
final_y = np.concatenate((y_ext_fp, middle_y, y_ext_fe))


# interpolate density function to the original proximity grid
f = interp1d(final_x, final_y, kind='linear', fill_value="extrapolate")
density_fit = f(prox)
np.save(os.path.join(path, 'density_fit.npy'), density_fit)

ind = np.argmin(np.abs(prox))
#print('density at prox=0, new fit vs original: ', density_fit[ind], density[ind])

# final plot showing the interpolated density profile
plt.plot(prox, density_fit, label='final interpolated fit')
plt.scatter(x_compressed, density, s=1, alpha=0.2, color='orange', label='compressed original data')
plt.axvline(x=0, color='black', linestyle='--')
plt.axvline(x=avg_lw/2, color='red', linestyle='--')
plt.axvline(x=-avg_lw/2, color='red', linestyle='--')
plt.xlabel('Proximity (Angstrom)')
plt.ylabel('Final Density (g/cm$^3$)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path, "final_density_fit.png"))
plt.close()

