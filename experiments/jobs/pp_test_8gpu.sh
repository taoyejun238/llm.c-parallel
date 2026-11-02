#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

# Pipeline-parallel correctness (d12, 12 layers). Runs that are compared see the same batches; the
# micro-batches of a step are the gradient accumulation steps (-d = B * T * dp * M):
# - PP 2/3/4/6 vs a single GPU, M = 4 or 8 micro-batches
# - PP with TP, CP, ZeRO-1 + DP

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

BIN=./bin/train_gpt2cu_cppp
DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/pp_test_8gpu
mkdir -p ${LOG}
COMMON="-e d12 -b 4 -t 1024 -x 30 -v 10000 -m 10 -s 0 -i ${DATA} -j ${DATA}"

compare() {  # file1 file2
  echo "----- loss diff $(basename $1 .out) vs $(basename $2 .out) -----"
  paste <(grep '^step' $1 | awk '{print $2, $5}') <(grep '^step' $2 | awk '{print $5}') \
    | awk 'BEGIN{m=0} {d=$3-$2; ad=d<0?-d:d; if(ad>m)m=ad; if(NR==1) printf "step 1: %s vs %s (%+.6f)\n", $2, $3, d} END{printf "%d steps, max |diff| = %.6f\n", NR, m}'
  echo "val: $(grep '^val loss' $1 | awk '{print $3}' | tr '\n' ' ')| $(grep '^val loss' $2 | awk '{print $3}' | tr '\n' ' ')"
}
run() {  # name ngpu args...
  NAME=$1; N=$2; shift 2
  echo "===== ${NAME}: -n ${N} $* ====="
  mpiexec -n ${N} ${BIN} ${COMMON} "$@" | tee ${LOG}/${NAME}.out
}

M4="-d 16384"   # 4 micro-batches of 4 x 1024 tokens
M8="-d 32768"   # 8 micro-batches
run base_m4 1 ${M4}
run pp2_m4 2 -pp 2 ${M4}
run pp4_m4 4 -pp 4 ${M4}
compare ${LOG}/base_m4.out ${LOG}/pp2_m4.out
compare ${LOG}/base_m4.out ${LOG}/pp4_m4.out

run base_m8 1 ${M8}
run pp3_m8 3 -pp 3 ${M8}
run pp6_m8 6 -pp 6 ${M8}
compare ${LOG}/base_m8.out ${LOG}/pp3_m8.out
compare ${LOG}/base_m8.out ${LOG}/pp6_m8.out

run pp2_tp2_m4 4 -pp 2 -tp 2 ${M4}
run pp2_cp2_m4 4 -pp 2 -cp 2 ${M4}
compare ${LOG}/base_m4.out ${LOG}/pp2_tp2_m4.out
compare ${LOG}/base_m4.out ${LOG}/pp2_cp2_m4.out

# DP 2 (total batch doubles): ZeRO-1 with PP vs plain DP
run base_dp2_z1_m4 2 -z 1 -d 32768
run pp4_dp2_z1_m4 8 -pp 4 -z 1 -d 32768
compare ${LOG}/base_dp2_z1_m4.out ${LOG}/pp4_dp2_z1_m4.out
