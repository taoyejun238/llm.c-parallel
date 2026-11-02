#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

# Pipeline parallelism vs tensor parallelism on one node: GPT-3 6.7B (32 layers), T=1024,
# 32 sequences per step (-d 32768), DP=1. M = number of micro-batches = 32 / B.

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

BIN=./bin/train_gpt2cu_cppp_perf
DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/pp_perf_8gpu
mkdir -p ${LOG}
COMMON="-e gpt3:c4096 -t 1024 -d 32768 -x 8 -v 10000 -m 1 -s 0 -i ${DATA} -j ${DATA}"

run() {  # name args...
  NAME=$1; shift
  echo "===== ${NAME}: $* ====="
  mpiexec -n 8 ${BIN} ${COMMON} "$@" 2>&1 | tee ${LOG}/${NAME}.out \
    | grep -E "memory per sequence|device memory usage|step    8|total average|out of memory|rror" | head -5
}

run tp8_b8 -tp 8 -b 8
run pp8_b4 -pp 8 -b 4
run pp8_b2 -pp 8 -b 2
run tp2_pp4_b4 -tp 2 -pp 4 -b 4
run tp4_pp2_b8 -tp 4 -pp 2 -b 8
