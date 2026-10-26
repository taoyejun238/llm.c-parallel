#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=6:00:00
#PJM -g ge45
#PJM -j

# Full baseline: GPT-2 124M on ~10B tokens of FineWeb, llm.c's recipe (scripts/run_gpt2_124M.sh):
# 18,865 steps of 524,288 tokens, lr 6e-4 with 700 warmup steps, weight decay 0.1, ZeRO-1,
# validation + HellaSwag every 250 steps. B=32 with 2 accumulation steps (B=64 does not fit in 40 GB).
# Checkpoints go outside the repository; resume with -y 1 if the job is interrupted.

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

OUT=/work/ge45/e45010/llmc_runs/gpt2_124M_fineweb
mkdir -p ${OUT}
mpiexec -n 8 ./bin/train_gpt2cu_tp \
  -i "dev/data/fineweb10B/fineweb_train_*.bin" \
  -j "dev/data/fineweb10B/fineweb_val_*.bin" \
  -o ${OUT} -v 250 -s 0 -h 1 -n 2000 -y 1 \
  -b 32 -t 1024 -d 524288 -r 0 -z 1 \
  -c 0.1 -l 0.0006 -q 0.0 -u 700 -x 18865 -e d12
cp ${OUT}/main.log experiments/logs/gpt2_124M_fineweb_main.log
