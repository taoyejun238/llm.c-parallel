#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=2:00:00
#PJM -g ge45
#PJM -j

# Longer loss-curve check of ZeRO-1/2/3 on GPT-2 124M / FineWeb: same recipe as the full baseline
# (llm.c's scripts/run_gpt2_124M.sh), shortened to 2000 steps (~1B tokens). B=32 with 2 accumulation
# steps instead of B=64, which does not fit in 40 GB.

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

for Z in 1 2 3; do
  OUT=experiments/logs/zero_curve_124M/z${Z}
  mkdir -p ${OUT}
  echo "===== d12 FineWeb 2000 steps zero_stage=${Z} ====="
  mpiexec -n 8 ./train_gpt2cu \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o ${OUT} -v 100 -s 0 -h 0 -n 0 \
    -b 32 -t 1024 -d 524288 -r 0 -z ${Z} \
    -c 0.1 -l 0.0006 -q 0.0 -u 700 -x 2000 -e d12
done
