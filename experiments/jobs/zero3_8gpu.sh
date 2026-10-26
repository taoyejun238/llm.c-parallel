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
LOG=experiments/logs/zero3_8gpu
mkdir -p ${LOG}
COMMON="-t 1024 -v 10000 -s 0 -i ${DATA} -j ${DATA}"

# print step, loss of stage $2, loss of stage $3, difference
compare() {
  echo "----- loss diff $1 (step, z$2, z$3, z$3-z$2) -----"
  paste <(grep '^step' ${LOG}/$1_z$2.out | awk '{print $2, $5}') \
        <(grep '^step' ${LOG}/$1_z$3.out | awk '{print $5}') \
    | awk '{printf "%s %s %s %+.6f\n", $1, $2, $3, $3 - $2}'
}

# 1) d12 B=16, no grad accumulation. ZeRO-1/2 rerun as a regression check against zero2_8gpu.
for Z in 1 2 3; do
  echo "===== d12 B=16 acc=1 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d12 -b 16 -x 50 -z ${Z} ${COMMON} | tee ${LOG}/d12_acc1_z${Z}.out
done
compare d12_acc1 2 3

# 2) d12 B=16, 2 grad accumulation steps
for Z in 2 3; do
  echo "===== d12 B=16 acc=2 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d12 -b 16 -d 262144 -x 20 -z ${Z} ${COMMON} | tee ${LOG}/d12_acc2_z${Z}.out
done
compare d12_acc2 2 3

# 3) d48 B=2: memory and speed
for Z in 2 3; do
  echo "===== d48 B=2 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d48 -b 2 -x 20 -z ${Z} ${COMMON} | tee ${LOG}/d48_z${Z}.out
done
compare d48 2 3
