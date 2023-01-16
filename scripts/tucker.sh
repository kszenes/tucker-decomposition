#!/bin/bash

FILENAME="/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_30_30_30.tns"
RANK=16

/users/kszenes/ParTI/tucker-decomp/build/tucker ${FILENAME} ${RANK} ${RANK} ${RANK}