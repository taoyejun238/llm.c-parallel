# llm.c-parallel

Distributed training experiments on top of [karpathy/llm.c](https://github.com/karpathy/llm.c) (C/CUDA), on an NVIDIA A100 GPU cluster. **Work in progress.**

## Goal

Implement and benchmark distributed training techniques in raw C/CUDA, and compare them against the llm.c baseline:

- ZeRO-2 (gradient sharding, overlapped with backward)
- ZeRO-3 (parameter sharding with per-layer all-gather and prefetching)
- Tensor Parallelism (Megatron-style, intra-node)
- Multi-dimensional parallelism across nodes (TP intra-node x ZeRO/DP inter-node)

Each implementation is validated in three steps: short loss-matching runs on 2 GPUs, a longer loss-curve comparison against the baseline on GPT-2 124M, and throughput/memory benchmarks.

## Environment

- GPU nodes with 8x NVIDIA A100-SXM4-40GB, NVLink intra-node, InfiniBand HDR inter-node
- CUDA 12.2, GCC 8.3.1, Open MPI 4.1.5 (CUDA-aware), NCCL 2.18.5

## Results so far

### Baseline scaling (weak scaling)

GPT-2 124M (`d12`), micro batch B=16 per GPU, T=1024, BF16, 50 steps. Numbers are taken from the stable steps at the end of each run.

| GPUs | Allocation | ms/step | tokens/s | MFU | Scaling efficiency |
|---|---|---|---|---|---|
| 1 | shared node | 114.7 | 142,760 | 36.8% | 100% |
| 2 | shared node | 116.9 | 280,200 | 36.0% | 98.1% |
| 4 | shared node | 117.7 | 556,420 | 35.8% | 97.4% |
| 8 (no ZeRO) | full node | 118.3 | 1,106,500 | 35.6% | 96.9% |
| 8 (ZeRO-1) | full node | 116.7 | 1,121,700 | 36.1% | 98.2% |

Notes:

- Scaling efficiency = throughput / (1-GPU throughput x number of GPUs).
- 1/2/4-GPU runs were on a shared node, where other jobs may run at the same time.
- ZeRO-1 is faster than plain data parallelism on 8 GPUs: the communication volume is the same (all-reduce = reduce-scatter + all-gather), but each GPU only runs the AdamW update on 1/8 of the parameters.

Raw logs: [`experiments/logs/`](experiments/logs/). Job scripts: [`experiments/jobs/`](experiments/jobs/).

### Memory ablation: ZeRO-1 and activation recomputation (8 GPUs)

GPT-2 1.5B (`d48`), micro batch B=2 per GPU, T=1024, BF16, 20 steps.

| ZeRO | Recompute | Memory / GPU | Est. max batch | ms/step | tokens/s | MFU |
|---|---|---|---|---|---|---|
| off | off | 35.0 GB | 3 | 211.9 | 77,350 | 30.4% |
| off | GeLU | 33.8 GB | 3 | 214.1 | 76,550 | 30.1% |
| ZeRO-1 | off | 19.4 GB | 6 | 180.8 | 90,550 | 35.6% |
| ZeRO-1 | GeLU | 18.2 GB | 7 | 182.9 | 89,560 | 35.2% |

- ZeRO-1 saves about 15.6 GB per GPU, matching the theoretical 7/8 x 12 bytes/param (FP32 master weights + Adam m, v) for 1.56B parameters, and roughly doubles the maximum micro batch.
- ZeRO-1 is also about 15% faster, since each GPU runs the AdamW update on only 1/8 of the parameters.
- Recomputing GeLU saves about 1.2 GB at B=2 for about 1% slowdown; the effect is small at this batch size.

### When the model does not fit: 2.7B without ZeRO

GPT-2 2.7B (`d60`, 2.75B parameters), B=1 per GPU, 8 GPUs.

| ZeRO | Memory / GPU | ms/step | tokens/s | MFU |
|---|---|---|---|---|
| off | 39.3 GB | 2,171 | 3,772 | 2.6% |
| ZeRO-1 | 22.3 GB | 196 | 41,630 | 28.8% |

Without ZeRO, the model states need about 44 GB per GPU (16 bytes/param), which exceeds the 40 GB of an A100. Instead of failing with OOM, llm.c falls back to `cudaMallocManaged` for the Adam m, v and master weights. The run still works, but every AdamW step pages about 33 GB over PCIe, making it about 11x slower. With ZeRO-1, all states fit in device memory.

## Changes to upstream llm.c

- `Makefile`: added a `NCCL_DIR` option so NCCL can be found on systems without `dpkg` (e.g. RHEL-based HPC clusters).
- `llmc/tokenizer.h`: fixed an uninitialized `eot_token` when the tokenizer file is missing, which made generation fail with "Token out of vocabulary". Submitted upstream as a [pull request](https://github.com/karpathy/llm.c/pulls?q=is%3Apr+author%3Ataoyejun238).

## Roadmap

- [x] Build llm.c on the cluster and run baseline scaling (1/2/4/8 GPUs)
- [ ] Full baseline: GPT-2 124M trained on 10B tokens of FineWeb, HellaSwag eval
- [x] Memory ablations on 1.5B (`d48`) with ZeRO-1 and activation recomputation, and a 2.7B (`d60`) fit test
- [ ] ZeRO-2
- [ ] ZeRO-3
- [ ] Tensor Parallelism
- [ ] Multi-node multi-dimensional parallelism on 7.3B+
- [ ] Profiling analysis with Nsight Systems
- Future work: Pipeline Parallelism (1F1B)

## Reproduce

Module names, paths and job scripts are specific to the cluster used; adapt them to your environment.

~~~bash
make train_gpt2cu GPU_COMPUTE_CAPABILITY=80 NCCL_DIR=/path/to/nccl OPENMPI_DIR=/path/to/openmpi

# data (any Python env with tiktoken, numpy, requests, tqdm, transformers)
python dev/data/tinyshakespeare.py

# 8-GPU baseline
mpiexec -n 8 ./train_gpt2cu -e d12 -b 16 -t 1024 -x 50 -v 10000 -s 0 -z 1 \
  -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
  -j dev/data/tinyshakespeare/tiny_shakespeare_train.bin
~~~

The original llm.c README is kept in [`README_llmc.md`](README_llmc.md).
