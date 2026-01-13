#!/bin/bash
#SBATCH --job-name=mtmb_test  # create a short name for your job
#SBATCH --nodes=1                # node count (this is a serial job)
#SBATCH --ntasks=1             # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=5:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin
#SBATCH --mail-type=end          # send email when job ends
##SBATCH --mail-user=

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2024.2
conda activate dp_plmd

module load cudatoolkit/12.4
module load cudnn/cuda-11.x/8.2.0
module load fftw/gcc/3.3.9
module load openmpi/gcc/4.1.2

module load /home/jl8178/.conda/envs/dp_plmd/lib/plumed/modulefile

lmp_plmd -in in.lammps