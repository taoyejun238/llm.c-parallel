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
COMMON="-t 1024 -x 20 -v 10000 -s 0 -i ${DATA} -j ${DATA}"

# d48 (1.5B): ZeRO x recompute ablation
for Z in 0 1; do
  for R in 0 1; do
    echo "===== d48 B=2 zero_stage=${Z} recompute=${R} ====="
    mpiexec -n 8 ./train_gpt2cu -e d48 -b 2 -z ${Z} -r ${R} ${COMMON}
  done
done

# d60 (2.7B): does it fit without ZeRO?
for Z in 0 1; do
  echo "===== d60 B=1 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d60 -b 1 -z ${Z} ${COMMON}
done
