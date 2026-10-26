#!/bin/bash
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=2:00:00
#PJM -g ge45
#PJM -j

cd /work/ge45/e45010/llm.c-parallel
export HF_HOME=/work/ge45/e45010/.cache/huggingface
export HF_DATASETS_OFFLINE=1
source /work/ge45/e45010/venvs/llmc/bin/activate
python dev/data/fineweb_local.py -v 10B
ls -lh dev/data/fineweb10B | head
