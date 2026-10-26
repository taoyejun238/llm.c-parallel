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
LOG=experiments/logs/zero2_8gpu
mkdir -p ${LOG}
COMMON="-t 1024 -v 10000 -s 0 -i ${DATA} -j ${DATA}"

# print step, z1 loss, z2 loss, z2 - z1
compare() {
  echo "----- loss diff $1 (step, z1, z2, z2-z1) -----"
  paste <(grep '^step' ${LOG}/$1_z1.out | awk '{print $2, $5}') \
        <(grep '^step' ${LOG}/$1_z2.out | awk '{print $5}') \
    | awk '{printf "%s %s %s %+.6f\n", $1, $2, $3, $3 - $2}'
}

# 1) d12 B=16, no grad accumulation: same setting as scale_8gpu.sh (loss match + speed)
for Z in 1 2; do
  echo "===== d12 B=16 acc=1 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d12 -b 16 -x 50 -z ${Z} ${COMMON} | tee ${LOG}/d12_acc1_z${Z}.out
done
compare d12_acc1

# 2) d12 B=16, 2 grad accumulation steps: tests per-micro-step accumulation into grads_shard
for Z in 1 2; do
  echo "===== d12 B=16 acc=2 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d12 -b 16 -d 262144 -x 20 -z ${Z} ${COMMON} | tee ${LOG}/d12_acc2_z${Z}.out
done
compare d12_acc2

# 3) d48 B=2: same setting as ablation_8gpu.sh (memory)
for Z in 1 2; do
  echo "===== d48 B=2 zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu -e d48 -b 2 -x 20 -z ${Z} ${COMMON} | tee ${LOG}/d48_z${Z}.out
done
compare d48
