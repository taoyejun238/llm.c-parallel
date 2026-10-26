#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

# Regression check: with -tp 1 the tensor-parallel build must reproduce the earlier ZeRO-1/2/3 runs
# (experiments/logs/zero3_8gpu/d12_acc1_z*.out) bit for bit.

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

BIN=./bin/train_gpt2cu_tp
DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/tp_regression_8gpu
mkdir -p ${LOG}
for Z in 1 2 3; do
  echo "===== d12 B=16 zero_stage=${Z} tp=1 ====="
  mpiexec -n 8 ${BIN} -e d12 -b 16 -t 1024 -x 50 -v 10000 -s 0 -z ${Z} -i ${DATA} -j ${DATA} | tee ${LOG}/d12_z${Z}.out
  paste <(grep '^step' experiments/logs/zero3_8gpu/d12_acc1_z${Z}.out | awk '{print $5}') \
        <(grep '^step' ${LOG}/d12_z${Z}.out | awk '{print $5}') \
    | awk -v z=${Z} 'BEGIN{m=0} {d=$2-$1; if(d<0)d=-d; if(d>m)m=d} END{printf "----- ZeRO-%d before vs after TP changes: %d steps, max |diff| = %.6f\n", z, NR, m}'
done
