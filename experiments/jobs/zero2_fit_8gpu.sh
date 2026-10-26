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
LOG=experiments/logs/zero2_fit_8gpu
mkdir -p ${LOG}
COMMON="-t 1024 -v 10000 -s 0 -i ${DATA} -j ${DATA}"

# d60 (2.7B): ZeRO-1 estimates max batch 3 at B=1; ZeRO-2 saves ~4 GB, about one extra sequence.
# B=4 should not fit in device memory with ZeRO-1 (falls back to managed memory), but fit with ZeRO-2.
for B in 1 4; do
  for Z in 1 2; do
    echo "===== d60 B=${B} zero_stage=${Z} ====="
    mpiexec -n 8 ./train_gpt2cu -e d60 -b ${B} -x 10 -z ${Z} ${COMMON} | tee ${LOG}/d60_b${B}_z${Z}.out
  done
done

# d72 (7.3B), B=1: ZeRO-1 model states alone are ~40 GB per GPU; ZeRO-2 needs ~28 GB.
for Z in 1 2; do
  echo "===== d72 B=1 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d72 -b 1 -x 5 -z ${Z} ${COMMON} | tee ${LOG}/d72_b1_z${Z}.out
done
