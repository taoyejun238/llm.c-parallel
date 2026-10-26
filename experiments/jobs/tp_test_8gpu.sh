#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

# Tensor-parallel correctness: keep the data-parallel size at 2 (so every run sees exactly the same
# batches) and vary the tensor-parallel size: 2 GPUs tp=1, 4 GPUs tp=2, 8 GPUs tp=4.
# Then combine tp=2 with ZeRO-1/2/3.

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

BIN=./bin/train_gpt2cu_tp
DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/tp_test_8gpu
mkdir -p ${LOG}
COMMON="-e d12 -b 8 -t 1024 -x 30 -v 10000 -s 0 -i ${DATA} -j ${DATA}"

compare() {
  echo "----- loss diff $1 vs $2 (step, $1, $2, diff) -----"
  paste <(grep '^step' ${LOG}/$1.out | awk '{print $2, $5}') <(grep '^step' ${LOG}/$2.out | awk '{print $5}') \
    | awk '{printf "%s %s %s %+.6f\n", $1, $2, $3, $3 - $2}'
}

for TP in 1 2 4; do
  N=$((2 * TP))
  echo "===== d12 B=8 dp=2 tp=${TP} zero_stage=0 ====="
  mpiexec -n ${N} ${BIN} ${COMMON} -tp ${TP} -z 0 | tee ${LOG}/tp${TP}_z0.out
done
compare tp1_z0 tp2_z0
compare tp1_z0 tp4_z0

for Z in 1 2 3; do
  echo "===== d12 B=8 dp=2 tp=2 zero_stage=${Z} ====="
  mpiexec -n 4 ${BIN} ${COMMON} -tp 2 -z ${Z} | tee ${LOG}/tp2_z${Z}.out
done
compare tp2_z0 tp2_z1
compare tp2_z1 tp2_z2
compare tp2_z2 tp2_z3
