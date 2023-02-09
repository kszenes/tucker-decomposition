#!/bin/bash

# === ORDER 3 ===

# FILENAME="/users/kszenes/ParTI/tucker-decomp/example_tensors/nell-2/nell-2.tns"
# RANK=16 

# FILENAME="/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_5_5_5.tns"
# RANK=4 

# /users/kszenes/ParTI/tucker-decomp/build/tucker ${FILENAME} ${RANK} ${RANK} ${RANK}

# === ORDER 4 ===

# FILENAME="/users/kszenes/ParTI/tucker-decomp/example_tensors/sparse_5_5_5_5.tns"
# RANK=4 
FILENAME="/users/kszenes/ParTI/tucker-decomp/example_tensors/enron/enron.tns"
RANK=16 

/users/kszenes/ParTI/tucker-decomp/build/tucker ${FILENAME} ${RANK} ${RANK} ${RANK} ${RANK}