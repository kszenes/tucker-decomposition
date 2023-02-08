#!/bin/bash

# FILENAME="/users/kszenes/ParTI/tucker-decomp/example_tensors/nell-2/nell-2.tns"
# RANK=16 

FILENAME="/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_5_5_5.tns"
RANK=5 

/users/kszenes/ParTI/tucker-decomp/build/tucker ${FILENAME} ${RANK} ${RANK} ${RANK}