#!/bin/bash
#
#SBATCH --job-name=mpi-amr
#
#SBATCH --time=48:00:00
#SBATCH --partition=aaiken
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

set -x

OUTDIR="mpi-output"
mkdir -p $OUTDIR

# Run a strong scaling experiment.

# Strong scaling.

# Run the serial algorithm.
# output_dir="$OUTDIR/strong/serial/"
# mkdir -p $output_dir
# srun -N 1 -n 1 --ntasks-per-node=1 ./serial-amr 10 30000 300 6 5 5 20 &> "$output_dir/output.out"

for n in 1 2; do
  for alg in "FINE_GRAIN"; do
    output_dir="$OUTDIR/strong/$alg/$n-node"
    mkdir -p $output_dir

    srun -N $n -n $n --ntasks-per-node=1 ./mpi-amr 10 30000 300 6 5 5 20 $alg $n &> "$output_dir/output.out"
  done
done
