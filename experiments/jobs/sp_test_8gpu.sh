#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

# Sequence-parallel correctness. Data-parallel size fixed at 2 so all runs see the same batches.
# - SP on/off at TP=2 and TP=4
# - SP with ZeRO-1/2/3
# - SP with 2 gradient accumulation steps (LayerNorm gradients are summed over the TP group per
#   micro-step with ZeRO-2/3, once on the last micro-step otherwise)
# - regression: without -sp, losses must be bit-identical to experiments/logs/tp_test_8gpu

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

BIN=./bin/train_gpt2cu_sp
DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/sp_test_8gpu
mkdir -p ${LOG}
COMMON="-e d12 -b 8 -t 1024 -x 30 -v 10000 -s 0 -i ${DATA} -j ${DATA}"

compare() {  # file1 file2
  echo "----- loss diff $(basename $1 .out) vs $(basename $2 .out) -----"
  paste <(grep '^step' $1 | awk '{print $2, $5}') <(grep '^step' $2 | awk '{print $5}') \
    | awk 'BEGIN{m=0} {d=$3-$2; ad=d<0?-d:d; if(ad>m)m=ad; if(NR==1) printf "step 1: %s vs %s (%+.6f)\n", $2, $3, d} END{printf "%d steps, max |diff| = %.6f\n", NR, m}'
}
run() {  # name ngpu args...
  NAME=$1; N=$2; shift 2
  echo "===== ${NAME}: -n ${N} $* ====="
  mpiexec -n ${N} ${BIN} ${COMMON} "$@" | tee ${LOG}/${NAME}.out
}

run tp2_sp0_z0 4 -tp 2 -z 0
run tp2_sp1_z0 4 -tp 2 -sp 1 -z 0
run tp4_sp0_z0 8 -tp 4 -z 0
run tp4_sp1_z0 8 -tp 4 -sp 1 -z 0
compare experiments/logs/tp_test_8gpu/tp2_z0.out ${LOG}/tp2_sp0_z0.out   # regression
compare experiments/logs/tp_test_8gpu/tp4_z0.out ${LOG}/tp4_sp0_z0.out   # regression
compare ${LOG}/tp2_sp0_z0.out ${LOG}/tp2_sp1_z0.out
compare ${LOG}/tp4_sp0_z0.out ${LOG}/tp4_sp1_z0.out
compare experiments/logs/tp_test_8gpu/tp1_z0.out ${LOG}/tp4_sp1_z0.out   # vs no TP at all

for Z in 1 2 3; do run tp2_sp1_z${Z} 4 -tp 2 -sp 1 -z ${Z}; done
compare ${LOG}/tp2_sp1_z0.out ${LOG}/tp2_sp1_z1.out
compare ${LOG}/tp2_sp1_z1.out ${LOG}/tp2_sp1_z2.out
compare ${LOG}/tp2_sp1_z2.out ${LOG}/tp2_sp1_z3.out

# 2 accumulation steps: total batch = 8 * 1024 * dp(2) * 2
run tp2_sp0_z1_acc2 4 -tp 2 -z 1 -d 32768
run tp2_sp1_z1_acc2 4 -tp 2 -sp 1 -z 1 -d 32768
run tp2_sp1_z2_acc2 4 -tp 2 -sp 1 -z 2 -d 32768
compare ${LOG}/tp2_sp0_z1_acc2.out ${LOG}/tp2_sp1_z1_acc2.out
compare ${LOG}/tp2_sp0_z1_acc2.out ${LOG}/tp2_sp1_z2_acc2.out
