#!/bin/bash
#PJM -L rscgrp=share-debug
#PJM -L elapse=0:10:00
#PJM -g ge45
#PJM -j

module purge
module load aquarius cuda/12.2 ompi-cuda/4.1.5-12.2
export LD_LIBRARY_PATH=/work/opt/local/x86_64/cores/nvidia/24.1/Linux_x86_64/24.1/comm_libs/nccl/lib:$LD_LIBRARY_PATH
cd /work/ge45/e45010/llm.c-parallel

DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin
mpiexec -n ${NGPU} ./train_gpt2cu -e d12 -b 16 -t 1024 -x 50 -v 10000 -s 0 \
  -i ${DATA} -j ${DATA} -o experiments/logs/scale_${NGPU}gpu
