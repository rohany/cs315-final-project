# CS315 Final Project

This project explored parallelization of an Adaptive Mesh Refinement (AMR) code in Regent, and compared
it to a serial and MPI based implementation.

Example parameters for sequential:
./serial-amr 10 1000 5 5 5 5 5

Example parameters for MPI:
./mpi-amr 10 1000 5 5 5 5 5

Example parameters for parallel:
./src/amr -i 10 -n 1000 -r 5 -rl 5 -rp 5 -rd 5 -ri 5 -ll:cpu N -ll:gpu N -ll:csize M
