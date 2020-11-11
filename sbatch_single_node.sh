#!/bin/bash
#
#SBATCH --job-name=single-node
#
#SBATCH --time=48:00:00
#SBATCH --partition=aaiken
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=4
#SBATCH --gres=gpu:4

set -x 

OUTDIR="single-node"
mkdir -p $OUTDIR

# Run the serial algorithm.
# output_dir="$OUTDIR/serial/"
# mkdir -p $output_dir
# ./serial-amr 10 15000 200 6 5 5 20 &> "$output_dir/output.out"
# 
# # Run the regent code.
# for n in 1 2 4 8; do
#   output_dir="$OUTDIR/regent/$n-core"
#   mkdir -p $output_dir
#   
#   ./src/amr -i 10 -n 15000 -r 200 -rl 6 -rp 5 -rd 5 -ri 20 -p $n -pr $n -ll:cpu $n -ll:csize 100000 -ll:util 2 &> "$output_dir/output.out"
# done

# Run the regent code with some GPUs.
for n in 1 2 4; do
  output_dir="$OUTDIR/regent/$n-gpu"
  mkdir -p $output_dir
  ./src/amr -i 10 -n 15000 -r 200 -rl 6 -rp 5 -rd 5 -ri 20 -p $(( n * 4 )) -pr $(( n * 4)) -ll:cpu 4 -ll:csize 64000 -ll:util 2 -ll:gpu $n -ll:fsize 15000 -ll:zsize 32000 &> "$output_dir/output.out"
done
