#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

# Context-parallel correctness (d12, T=1024). Runs that are compared see the same batches:
# - CP 2/4/8 vs a single GPU (data-parallel size 1)
# - CP with TP, SP, ZeRO-1/2/3
# - CP 4 x DP 2 vs DP 2

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

BIN=./bin/train_gpt2cu_cppp
DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
LOG=experiments/logs/cp_test_8gpu
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

run base_dp1 1
run cp2 2 -cp 2
run cp4 4 -cp 4
run cp8 8 -cp 8
compare ${LOG}/base_dp1.out ${LOG}/cp2.out
compare ${LOG}/base_dp1.out ${LOG}/cp4.out
compare ${LOG}/base_dp1.out ${LOG}/cp8.out

run cp2_tp2 4 -cp 2 -tp 2
run cp2_tp2_sp 4 -cp 2 -tp 2 -sp 1
compare ${LOG}/base_dp1.out ${LOG}/cp2_tp2.out
compare ${LOG}/cp2_tp2.out ${LOG}/cp2_tp2_sp.out

for Z in 1 2 3; do run cp2_z${Z} 2 -cp 2 -z ${Z}; done
compare ${LOG}/cp2.out ${LOG}/cp2_z1.out
compare ${LOG}/cp2_z1.out ${LOG}/cp2_z2.out
compare ${LOG}/cp2_z2.out ${LOG}/cp2_z3.out

run base_dp2 2
run cp4_dp2 8 -cp 4
compare ${LOG}/base_dp2.out ${LOG}/cp4_dp2.out
