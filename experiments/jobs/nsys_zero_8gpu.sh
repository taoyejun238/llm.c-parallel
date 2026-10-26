#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:30:00
#PJM -g ge45
#PJM -j

# Nsight Systems traces of ZeRO-1/2/3 on GPT-2 1.5B (d48), B=2, 8 GPUs. Only rank 0 is profiled.

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
OUT=experiments/profiles
mkdir -p ${OUT}

for Z in 1 2 3; do
  echo "===== nsys d48 B=2 zero_stage=${Z} ====="
  mpiexec -n 8 bash -c '
    if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
      exec nsys profile --trace=cuda,nvtx --force-overwrite=true -o '"${OUT}"'/d48_z'"${Z}"'_rank0 "$@"
    else
      exec "$@"
    fi' _ ./train_gpt2cu -e d48 -b 2 -t 1024 -x 8 -v 10000 -s 0 -z ${Z} -i ${DATA} -j ${DATA}
done
