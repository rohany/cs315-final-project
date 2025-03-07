#!/bin/bash
#
#SBATCH --job-name=regent-amr
#
#SBATCH --time=48:00:00
#SBATCH --partition=aaiken
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

# cd src/
# make amr
# cd ../

set -x

OUTDIR="regent-output"
mkdir -p $OUTDIR

# Run a strong scaling experiment on just 1 core on each node.

for n in 2; do
  output_dir="$OUTDIR/1-core/strong/$n-node"
  mkdir -p $output_dir

  srun -N $n -n $n --ntasks-per-node=1 ./src/amr -i 10 -n 30000 -r 300 -rl 6 -rp 5 -rd 5 -ri 20 -p $n -pr $n -ll:cpu 1 -ll:csize 100000 -ll:util 2 &> "$output_dir/output.out"
done

# Run a strong scaling experiment using more cores on each node.

for n in 2; do
  output_dir="$OUTDIR/8-cores/strong/$n-node"
  mkdir -p $output_dir

  srun -N $n -n $n --ntasks-per-node=1 ./src/amr -i 10 -n 30000 -r 300 -rl 6 -rp 5 -rd 5 -ri 20 -p $(( n * 8)) -pr $(( n * 8)) -ll:cpu 8 -ll:csize 100000 -ll:util 2 &> "$output_dir/output.out"
done

# Strong scale with # of GPUs first.

# TODO (rohany): Run a strong scaling experiment using alot of GPU's.
# for n in 3 4; do
#   output_dir="$OUTDIR/strong-gpu/$n-gpus"
#   mkdir -p $output_dir
# 
#   srun -N 1 -n 1 --ntasks-per-node=1 --gpus-per-task=$n ./src/amr -i 10 -n 30000 -r 12000 -rl 0 -rp 5 -rd 5 -ri 20 -p $(( 2 * n )) -pr $(( 2 * n )) -ll:cpu 8 -ll:csize 64000 -ll:util 2 -ll:fsize 15000 -ll:zsize 32000 -ll:gpu $n &> "$output_dir/output.out"
# done
