#!/bin/bash

FILENAME="/users/kszenes/ParTI/tucker-decomp/example_tensors/nell-2/nell-2.tns"
RANK=16 

/users/kszenes/ParTI/tucker-decomp/build/tucker ${FILENAME} ${RANK} ${RANK} ${RANK}