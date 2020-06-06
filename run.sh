#!/bin/bash

# clone hypre from hypre-space
git clone --branch v2.19.0 https://github.com/hypre-space/hypre.git

# build hypre, required mpi
cd hypre/src/
./configure
make -j 4 install
cd ../..

# build test driver for MGR
make driver

# extract input files
tar zxf input.tar.gz

# run the solver on the example matrix using the preconditioner and rhs
./driver -fromfile full_mat -precondfromfile full_precond -rhsfromfile full_rhs -blockCF blockIdx
