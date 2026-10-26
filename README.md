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

### ZeRO-2 and ZeRO-3

Model state per GPU with N GPUs (bytes per parameter):

| | BF16 params | BF16 grads | FP32 master + Adam m, v | Total (N=8) |
|---|---|---|---|---|
| ZeRO-1 (upstream) | 2 | 2 | 12/N | 5.5 |
| ZeRO-2 | 2 | 2/N | 12/N | 3.75 |
| ZeRO-3 | 2/N | 2/N | 12/N | 2.0 |

How they work here:

- **ZeRO-2** ([`llmc/zero2.cuh`](llmc/zero2.cuh)): each layer's gradients are computed into one of two reusable layer buffers. Right after the layer's backward, they are reduce-scattered on the NCCL stream, and this rank's 1/N is accumulated into a gradient shard, while the compute stream moves on to the next layer with the other buffer. CUDA events order the two streams in both directions. The shard has the same layout as the sharded Adam m/v, so the optimizer reads it directly.
- **ZeRO-3** ([`llmc/zero3.cuh`](llmc/zero3.cuh)): the 10 large per-layer tensors (`qkvw` ... `fcprojb`) are kept only as a 1/N shard. Before a layer runs in forward or backward, its parameters are all-gathered into one of two layer buffers, and the next layer is prefetched while the current one computes. AdamW updates the shard in place, so there is no all-gather after the update. Gradients go through the ZeRO-2 path.
- `wte`, `wpe`, `lnf` (and `ln1` for ZeRO-3) keep full-size buffers: `wte` gets gradient at both ends of the backward pass, and `ln1` of layer l+1 is used inside layer l by a fused kernel.

**Correctness** (8 GPUs, tinyshakespeare):

- ZeRO-2 vs ZeRO-1, `d12` B=16, 50 steps: identical for the first 6 steps, below 0.0003 until step 30, at most 0.051 afterwards. This is floating-point drift amplified by gradient-norm spikes; the original ZeRO-0 vs ZeRO-1 code drifts as much (0.061) on the same run. With 2 gradient accumulation steps (20 steps): at most 0.0037. `d48`, 20 steps: at most 0.0012.
- ZeRO-3 vs ZeRO-2: **bit-identical** losses on every run where both fit (`d12` 50 steps, `d12` with gradient accumulation, `d48`, `d72`).
- ZeRO-1 and ZeRO-2 losses are bit-identical before and after each change.

**Memory and speed at the same batch** (memory = device memory per GPU, measured):

| Model | B | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| 124M (`d12`) | 16 | 11.9 GB, 116.9 ms | 11.8 GB, 116.8 ms | 11.7 GB, 119.3 ms |
| 1.5B (`d48`) | 2 | 18.2 GB, 182.9 ms | 15.9 GB, 183.8 ms | 13.6 GB, 189.3 ms |
| 2.7B (`d60`) | 1 | 22.3 GB, 197.0 ms | 18.0 GB, 198.8 ms | not run |
| 7.3B (`d72`) | 1 | does not fit, 1,915 ms (Adam state in host memory) | 37.6 GB, 417.8 ms | 26.0 GB, 443.7 ms |

**What fits on 8x A100 40GB** (tokens/s, higher is better):

| Model | B | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| 2.7B (`d60`) | 4 | OOM | **60,830** (37.4 GB) | not run |
| 7.3B (`d72`) | 1 | 4,277 (host memory) | **19,606** | 18,461 |
| 7.3B (`d72`) | 2 | not run | 9,013 (host memory) | **24,941** (35.4 GB) |
| 12.2B (`d84`) | 1 | not run | OOM | **11,755** (39.5 GB) |

- ZeRO-2 saves 2.3 GB on 1.5B and 4.2 GB on 2.7B. On 2.7B that is enough for one more sequence: B=4 runs at 48% higher throughput than B=1, while ZeRO-1 runs out of memory.
- ZeRO-3 saves another 11.5 GB on 7.3B. It runs 7.3B at B=2 (2.8x faster than ZeRO-2, which has to page Adam state over PCIe) and 12.2B, which ZeRO-2 cannot run at all.
- Cost: without gradient accumulation ZeRO-2 is as fast as ZeRO-1. With 2 accumulation steps it is 0.7% slower, because it reduce-scatters every micro-step (ZeRO-1 reduces once). ZeRO-3 all-gathers each layer twice per micro-step and is 3-6% slower than ZeRO-2 at the same batch on one NVLink node.

Limitations:

- Only short runs so far (5-50 steps). The longer loss-curve comparison on GPT-2 124M is still to do.
- ZeRO-3 does not support writing or resuming from checkpoints yet (it exits with an error if `-n` or `-y 1` is set).
- The `d84` run is unstable over its 5 steps (loss 11 to 25). The same spikes appear identically with ZeRO-2 and ZeRO-3 on `d72`, so this is most likely the learning rate (3e-4, no warmup) at this scale, but it cannot be compared against ZeRO-2 because ZeRO-2 does not fit.

Logs: `experiments/logs/zero2_8gpu*`, `experiments/logs/zero3_8gpu*`, `experiments/logs/zero2_fit_8gpu*`, `experiments/logs/zero3_fit_8gpu*`.

## Changes to upstream llm.c

- `Makefile`: added a `NCCL_DIR` option so NCCL can be found on systems without `dpkg` (e.g. RHEL-based HPC clusters).
- `llmc/tokenizer.h`: fixed an uninitialized `eot_token` when the tokenizer file is missing, which made generation fail with "Token out of vocabulary". Submitted upstream as a [pull request](https://github.com/karpathy/llm.c/pulls?q=is%3Apr+author%3Ataoyejun238).
- ZeRO-2 and ZeRO-3: new `llmc/zero2.cuh` and `llmc/zero3.cuh`; `llmc/zero.cuh` accepts `-z 2` and `-z 3`; `train_gpt2.cu` changes gradient/parameter allocation, the per-layer pointers in forward and backward, the gradient norm and the AdamW update.
- `train_gpt2.cu`: the all-gather after the AdamW update now waits for the update on the compute stream (event record + wait). Upstream starts it on the NCCL stream without this dependency.

## Roadmap

- [x] Build llm.c on the cluster and run baseline scaling (1/2/4/8 GPUs)
- [ ] Full baseline: GPT-2 124M trained on 10B tokens of FineWeb, HellaSwag eval
- [x] Memory ablations on 1.5B (`d48`) with ZeRO-1 and activation recomputation, and a 2.7B (`d60`) fit test
- [x] ZeRO-2 (short-run loss matching, memory and throughput on 1.5B-7.3B)
- [x] ZeRO-3 (bit-identical to ZeRO-2; trains 7.3B at B=2 and 12.2B on one 8-GPU node)
- [ ] Longer loss-curve validation of ZeRO-2/3 on GPT-2 124M
- [ ] ZeRO-3 checkpoint writing and resuming
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

# ZeRO-2 / ZeRO-3: same command with -z 2 or -z 3 (needs more than one GPU; ZeRO-3 needs -n 0)
mpiexec -n 8 ./train_gpt2cu -e d72 -b 2 -t 1024 -x 5 -v 10000 -s 0 -z 3 \
  -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
  -j dev/data/tinyshakespeare/tiny_shakespeare_train.bin
~~~

The original llm.c README is kept in [`README_llmc.md`](README_llmc.md).
