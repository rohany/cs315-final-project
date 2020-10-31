#!/bin/bash
#
#SBATCH --job-name=regent-amr
#
#SBATCH --time=48:00:00
#SBATCH --partition=aaiken
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9

srun -N 1 -n 1 --ntasks-per-node=1 ./src/amr -i 10 -n 1000 -r 5 -rl 5 -rp 5 -rd 5 -ri 5 -p 8 -ll:cpu 2 -ll:csize 4000
srun -N 2 -n 2 --ntasks-per-node=1 ./src/amr -i 10 -n 1000 -r 5 -rl 5 -rp 5 -rd 5 -ri 5 -p 8 -ll:cpu 2 -ll:csize 4000
