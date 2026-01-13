#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025
PURPOSE: Extract and/or modify Ne atom data from rawdata.npz
Step 2 (after extract.py) in the data post-processing and analysis pipeline.

INPUT:
    - rawdata.npz
    - frame.txt
    - mod.txt (if any Ne proximity coordinate needs to be adjusted with -m 1)

OUTPUT:
    - ne.txt
    - indices_mgo.txt (optional)
    - Ne_prox_scatter.png

    (if -n <N> <id> is specified)
    - atom_type_mod.npy
    - Ne_prox_scatter_rm.png

    (if -m 1 is specified)
    - ne.npz
    - Ne_prox_scatter_mod.png
    - Ne_phase_mod.png

Example usage: 
First run "python ne.py" without any flags to observe raw Ne data in itself
Then, run "python ne.py -m 1" and "python ne.py -n 10 5 8" with flags to modify the Ne data accordingly
- redefines proximity and phase data based on mod.txt, which specifies which Ne atoms need adjustment at which frames
- extracts info on Ne atoms 5 and 8 (1-indexed) that are trapped in the ferropericlase phase, out of 10 Ne atoms total
- Ne atom IDs can be identified by looking at Ne_prox_scatter.png, where the trajectory colors correspond to the
matplotlib tab10 color cycle. A purple trajectory corresponds to Ne atom 5 (or 15, if the total number of Ne is >=15).
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str, default='./', help="working directory")
parser.add_argument("--mod", "-m", type=int, default=0, help="modify raw Ne data to wrap around proximity bounds")
parser.add_argument("--ne", "-n", nargs='+', type=int, help="format list: <total Ne> <trapped Ne IDs>")

args = parser.parse_args()
path = args.path
mod = args.mod

# text file containing start frame and end frame of the portion of the trajectory analyzed
frame_file = os.path.join(path, "frame.txt")
if not os.path.isfile(frame_file):
    print(f"File {frame_file} not found")
    exit()

start_frame = 0
end_frame = 0
with open(frame_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]
    start_frame = int(lines[0])
    end_frame = int(lines[1])

# access raw data generated from extract.py
raw = np.load(os.path.join(path, 'rawdata.npz'))

Ne = raw['Ne']
lw = raw['lw']

atom_type = raw['atom_type']
atom_prox = raw['atom_prox']
atom_phase = raw['atom_phase']


# for Ne atoms that are trapped in the MgO phase and do not participate in elemental partitioning
if args.ne:
    # outputs modified atom_type array where the trapped Ne atoms are re-labeled as type 4 instead of type 3
    start = int(len(atom_type)-args.ne[0])
    ne = args.ne[1:]
    for atom in ne:
        atom_type[start+atom-1] = 4 # trapped Ne
    np.save("atom_type_mod.npy", atom_type)

    time = np.array(range(len(Ne)))+start_frame/500

    # time evolution of position along proximity axis for each Ne atom, after removing trapped Ne atoms
    plt.figure(figsize=(10, 6))
    for i in range(len(atom_type)):
        if atom_type[i]==3: # normal Ne atoms
            plt.scatter(time,atom_prox[:,i],marker='.',alpha=0.4)
    plt.ylabel('Proximity coordinate')
    plt.xlabel('Time index')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "Ne_prox_scatter_rm.png"))
    plt.close()
    exit()

# for any frames where Ne atom proximity coordinate is recorded to be erroneously far from the interface
if mod:
    if not os.path.isfile(os.path.join(path,'mod.txt')):
        print(f"File mod.txt not found")
        exit()
    else:
        # mod.txt file format: 
        # <total number of Ne atoms>
        # <atom ID (1-indexed)> <frame number 1> <frame number 2> ...
        # <atom ID (1-indexed)> <frame number 1> <frame number 2> ...
        with open(os.path.join(path, 'mod.txt'), 'r') as f:
            mod_data = [list(map(int, line.strip().split())) for line in f if line.strip()]

        atom_list = []
        idx_list = []
        natoms = range(int(len(atom_type)-mod_data[0][0]),len(atom_type)) # indices of Ne atoms in atom_type array

        # fill atom_list with Ne atom indices and idx_list with corresponding lists of frame indices to be modified
        for i in range(1, len(mod_data)):
            atom_list.append(natoms[int(mod_data[i][0]-1)])
            idx = np.array(mod_data[i][1:])
            idx_list.append((idx-int(start_frame/500)).astype(int))
            
        for atom, idx in zip(atom_list, idx_list):
            for i in idx:
                print(atom_prox[i,atom]) # proximity coordinate of the Ne atom at frame i (subject to modification)
                print(atom_prox[i-2:i+3,atom]) # proximity coordinates of the Ne atom at frames i-2 to i+2 (for comparison)

                # the hope is that the Ne atom in frame i is an aberration and the surrounding frames capture the actual
                # proximity to an interface. this can be assessed from the printed output above.
                # this rudimentary algorithm takes the surrounding frames (that are not in idx, which represent aberrations)
                # and replaces the proximity coordinate at frame i with the average of the proximity coordinates from the
                # two closest "normal" frames. this is not perfect and obtains the new proximity coordinate from rough
                # interpolation. getting an accurate proximity coordinate with a more robust algorithm may require accessing
                # the original trajectory and using tools like MDAnalysis. for now, this works ok.
                lower = i-1
                upper = i+1
                while lower in set(idx):
                    lower -=1
                while upper in set(idx):
                    upper +=1
                prox_new = (atom_prox[lower,atom]+atom_prox[upper,atom])/2
                print('->',prox_new)

                # update proximity coordinate and phase of the Ne atom at frame i based on new proximity coordinate
                # for atom_phase, 2 is liquid, 1 is solid, 0 is interface
                # for Ne array, the three counts are in the order of [liquid, solid, interface]
                if prox_new<lw[i]/2 and prox_new>-lw[i]/2: # interface region
                    if atom_prox[i,atom] < -lw[i]/2:
                        # atom originally recorded to be in solid MgO phase
                        atom_prox[i,atom] = prox_new
                        atom_phase[i,atom] = 0
                        Ne[i,1] -= 1 # takes form [liquid, solid, interface]
                        Ne[i,2] += 1
                        print('now interface')
                    else:
                        # atom originally recorded to be in liquid Fe phase
                        atom_prox[i,atom] = prox_new
                        atom_phase[i,atom]=0 
                        Ne[i,0] -= 1 # takes form [liquid, solid, interface]
                        Ne[i,2] += 1
                        print('now interface')
                elif prox_new<=-lw[i]/2 and atom_phase[i,atom]==2:
                    # atom originally recorded to be in liquid Fe phase but actually in solid MgO phase
                    atom_prox[i,atom] = prox_new
                    atom_phase[i,atom]=1
                    Ne[i,0] -= 1
                    Ne[i,1] += 1
                    print('now solid')
    # adjustments saved in ne.npz
    np.savez_compressed(os.path.join(path,'ne.npz'), atom_prox=atom_prox, atom_phase=atom_phase, Ne=Ne)

    time = np.array(range(len(Ne)))+start_frame/500

    # time evolution of position along proximity axis for each Ne atom, after modifying anomalous proximity coordinates
    plt.figure(figsize=(10, 6))
    for i in range(len(atom_type)):
        if atom_type[i]==3: # Ne
            plt.scatter(time,atom_prox[:,i],marker='.',alpha=0.4)
    plt.ylabel('Proximity coordinate')
    plt.xlabel('Time index')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "Ne_prox_scatter_mod.png"))
    plt.close()

    # time evolution of phase occupancy for each Ne atom, after modifying anomalous proximity coordinates
    plt.figure(figsize=(10, 6))
    for i in range(len(atom_type)):
        if atom_type[i]==3: # Ne
            phase = []
            for j in range(len(Ne)):
                if atom_phase[j][i] == 2:
                    phase.append(1)
                elif atom_phase[j][i] == 1:
                    phase.append(-1)
                else:
                    phase.append(atom_phase[j][i])
            plt.plot(time, phase, alpha=0.8)
    plt.ylabel('Atom Phase (-1: solid, 0: interface, 1: liquid)')
    plt.xlabel('Frame Index')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "Ne_phase_mod.png"))
    plt.close()
    exit()


# records every frame where Ne atoms are present in the liquid Fe phase
# helpful to double check this output with visual inspection of the molecular trajectory
# to catch errors or inform contents of mod.txt
with open(os.path.join(path,'ne.txt'), 'w') as f:
    for i in range(len(Ne)):
        if Ne[i,0] > 0:
            f.write(f"{i+int(start_frame/500)}   {int(Ne[i,0])}    {start_frame+i*500}\n")
f.close()

time = np.array(range(len(Ne)))+start_frame/500

# time evolution of position along proximity axis for each Ne atom
# similar to Ne_prox.png generated from extract.py
plt.figure(figsize=(10, 6))
for i in range(len(atom_type)):
    if atom_type[i]==3: # Ne
        plt.scatter(time,atom_prox[:,i],marker='.',alpha=0.4)

        # output indices_mgo.txt can be utilized and modified as needed
        # currently records frames where Ne is located more than 10 Angstroms away from the interface inside MgO phase
        # due to periodic boundary conditions, positions far away from the interface plane can actually wrap back around
        # and be close to an interface from the other side, but this may not be captured correctly every time with the
        # GDS analysis and needs this extra screening. visual inspection is recommended to verify these results.
        # this information provides content for mod.txt if proximity of Ne atoms needs to be adjusted.
        # it is also helpful to do this for Ne atoms found far inside the Fe phase (e.g., >7 Angstroms from interface)
        indices = np.where((atom_prox[:,i] < -10))[0]
        #indices = np.where((atom_prox[:,i] > 7))[0]
        print(len(indices))
        if len(indices) > 300:
            # approximate way of skipping Ne atoms that are trapped and reside deep in the MgO phase
            continue
        elif len(indices) > 0:
            with open(os.path.join(path, f"indices_mgo.txt"), "a") as idx_file:
                for idx in indices:
                    idx_file.write(f"{int(start_frame/500)+idx} ") # {atom_prox[idx,i]}\n")
                idx_file.write("\n")

plt.ylabel('Proximity coordinate')
plt.xlabel('Time index')
plt.tight_layout()
plt.savefig(os.path.join(path, "Ne_prox_scatter.png"))
plt.close()
