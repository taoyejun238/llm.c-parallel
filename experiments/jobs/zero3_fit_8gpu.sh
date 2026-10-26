#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/zero3_fit_8gpu
mkdir -p ${LOG}
COMMON="-t 1024 -v 10000 -s 0 -i ${DATA} -j ${DATA}"

# d72 (7.3B): ZeRO-2 fits B=1 (37.6 GB) but not B=2; ZeRO-3 frees ~12 GB of parameters per GPU.
echo "===== d72 B=1 zero_stage=3 ====="
mpiexec -n 8 ./train_gpt2cu -e d72 -b 1 -x 5 -z 3 ${COMMON} | tee ${LOG}/d72_b1_z3.out
for Z in 2 3; do
  echo "===== d72 B=2 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d72 -b 2 -x 5 -z ${Z} ${COMMON} | tee ${LOG}/d72_b2_z${Z}.out
done

# d84 (12.2B): bf16 parameters alone are 24 GB per GPU without parameter sharding.
for Z in 2 3; do
  echo "===== d84 B=1 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d84 -b 1 -x 5 -z ${Z} ${COMMON} | tee ${LOG}/d84_b1_z${Z}.out
done
