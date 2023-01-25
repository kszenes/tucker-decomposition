#!/bin/bash

FILENAME="/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_10_10_10.tns"
RANK=10 

/users/kszenes/ParTI/tucker-decomp/build/tucker ${FILENAME} ${RANK} ${RANK} ${RANK}