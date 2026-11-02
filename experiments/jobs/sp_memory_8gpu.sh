#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

# Sequence-parallel memory/throughput: GPT-3 6.7B (32 heads), TP=8 on one node (DP=1), SP off vs on.

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

BIN=./bin/train_gpt2cu_sp
DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/sp_memory_8gpu
mkdir -p ${LOG}
COMMON="-e gpt3:c4096 -t 1024 -x 8 -v 10000 -s 0 -tp 8 -i ${DATA} -j ${DATA}"

run() {  # name args...
  NAME=$1; shift
  echo "===== ${NAME}: $* ====="
  mpiexec -n 8 ${BIN} ${COMMON} "$@" | tee ${LOG}/${NAME}.out
}

run c4096_tp8_sp0_b8  -b 8
run c4096_tp8_sp1_b8  -b 8 -sp 1
run c4096_tp8_sp0_b12 -b 12
run c4096_tp8_sp1_b12 -b 12 -sp 1
run c4096_tp8_sp1_b16 -b 16 -sp 1
