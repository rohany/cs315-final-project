#!/bin/bash
#
#SBATCH --job-name=mpi-amr
#
#SBATCH --time=48:00:00
#SBATCH --partition=aaiken
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9

srun -N 1 -n 1 --ntasks-per-node=1 ./mpi-amr 10 1000 5 5 5 5 5 FINE_GRAIN 
srun -N 2 -n 2 --ntasks-per-node=1 ./mpi-amr 10 1000 5 5 5 5 5 FINE_GRAIN
