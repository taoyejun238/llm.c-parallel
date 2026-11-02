#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

# Context parallelism for long sequences: d12 (124M), one sequence per step (B=1), T = 4096 and 16384.
# The (non-flash) attention stores a T x T matrix per head and layer, so memory grows with T^2;
# CP splits the queries of every sequence over the CP ranks.

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

BIN=./bin/train_gpt2cu_cppp_perf
DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/cp_long_8gpu
mkdir -p ${LOG}
COMMON="-e d12 -b 1 -x 10 -v 10000 -m 1 -s 0 -i ${DATA} -j ${DATA}"

run() {  # name ngpu args...
  NAME=$1; N=$2; shift 2
  echo "===== ${NAME}: -n ${N} $* ====="
  mpiexec -n ${N} ${BIN} ${COMMON} "$@" 2>&1 | tee ${LOG}/${NAME}.out \
    | grep -E "memory per sequence|device memory usage|step   10|out of memory|rror" | head -5
}

for CP in 1 2 4 8; do run t4096_cp${CP} ${CP} -t 4096 -cp ${CP}; done
for CP in 2 4 8; do run t16384_cp${CP} ${CP} -t 16384 -cp ${CP}; done
