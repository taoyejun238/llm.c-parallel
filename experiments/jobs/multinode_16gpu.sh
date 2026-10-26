#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=2
#PJM -L elapse=2:00:00
#PJM -g ge45
#PJM -j

# 2 nodes x 8 A100 (NVLink inside a node, InfiniBand between nodes).
# Tensor parallelism inside a node x data parallelism (with ZeRO) across nodes, vs pure ZeRO-3.
# Ranks are placed 8 per node in order, so a TP group of consecutive ranks never crosses nodes.

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

BIN=./bin/train_gpt2cu_tp
DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/multinode_16gpu
mkdir -p ${LOG}
COMMON="-t 1024 -v 10000 -s 0 -i ${DATA} -j ${DATA}"
RUN="mpiexec -machinefile ${PJM_O_NODEINF} -n 16 -npernode 8"

run() {  # name, args...
  NAME=$1; shift
  echo "===== $NAME: $* ====="
  ${RUN} ${BIN} ${COMMON} "$@" | tee ${LOG}/${NAME}.out
}

# smoke test + 2-node weak scaling of plain data parallelism (compare with 8-GPU d12 B=16: 116.7 ms)
run d12_dp16_z1 -e d12 -b 16 -x 30 -z 1

# GPT-3 6.7B (32 heads): TP=8 inside each node x DP=2 across nodes, vs ZeRO-3 over all 16 GPUs
run c4096_tp8_dp2_z1_b8  -e gpt3:c4096 -b 8  -x 10 -tp 8 -z 1
run c4096_tp8_dp2_z1_b16 -e gpt3:c4096 -b 16 -x 10 -tp 8 -z 1
run c4096_dp16_z3_b1     -e gpt3:c4096 -b 1  -x 10 -z 3
run c4096_dp16_z3_b4     -e gpt3:c4096 -b 4  -x 10 -z 3

# GPT-2 12.2B (d84, 36 heads): TP=4 x DP=4, vs ZeRO-3 over all 16 GPUs
run d84_tp4_dp4_z1_b2 -e d84 -b 2 -x 10 -tp 4 -z 1
run d84_tp4_dp4_z3_b4 -e d84 -b 4 -x 10 -tp 4 -z 3
run d84_dp16_z3_b1    -e d84 -b 1 -x 10 -z 3
