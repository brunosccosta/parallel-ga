parallel-ga
===========

Parallel implementation of Genetic Algorithms using Pthread, CUDA and MPI.

Four versions of this algorithm. Written for educational and testing purposes
- ea.c: Basic version, fully sequential, usefull to compare times.
- ea-thread.c: Multithread version using Pthread library.
- ea.cu/rand.cu: CUDA kernels for CUDA versions.
- ea-mpi.c: Distributed version using MPI.
