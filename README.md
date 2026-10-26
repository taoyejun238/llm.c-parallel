# llm.c-parallel

Distributed training experiments on top of [karpathy/llm.c](https://github.com/karpathy/llm.c) (C/CUDA), on an NVIDIA A100 GPU cluster: ZeRO-2, ZeRO-3 and Megatron-style tensor parallelism implemented in raw C/CUDA + NCCL, validated against the llm.c baseline, and combined across nodes.

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

## Results

### Full baseline: GPT-2 124M on 10B tokens of FineWeb

llm.c's recipe ([`scripts/run_gpt2_124M.sh`](scripts/run_gpt2_124M.sh)) on one 8-GPU node with ZeRO-1: 18,865 steps of 524,288 tokens (9.9B tokens), lr 6e-4 with 700 warmup steps and cosine decay to 0, weight decay 0.1. B=32 with 2 gradient accumulation steps instead of B=64, which does not fit in 40 GB.

| Step | 1,000 | 2,000 | 5,000 | 10,000 | 15,000 | 18,865 (end) |
|---|---|---|---|---|---|---|
| Val loss | 4.332 | 3.869 | 3.566 | 3.401 | 3.310 | **3.288** |
| HellaSwag (acc_norm) | 25.4% | 25.9% | 27.4% | 29.0% | 29.7% | **29.8%** |

- HellaSwag ends at 29.8% (best 30.0% at step 16,750), above OpenAI's GPT-2 124M at **29.55%**, measured with the same eval script ([`dev/data/hellaswag.py`](dev/data/hellaswag.py), log in `experiments/logs/hellaswag_openai_gpt2_124M.log`).
- 2.8 hours, about 1.0M tokens/s, 34% MFU.
- Log: [`experiments/logs/gpt2_124M_fineweb_main.log`](experiments/logs/gpt2_124M_fineweb_main.log) (`tel` = val loss, `eval` = HellaSwag). FineWeb was tokenized from a local copy of the 10BT sample with [`dev/data/fineweb_local.py`](dev/data/fineweb_local.py) (`dev/data/fineweb.py` with a local parquet path).

### Baseline scaling (weak scaling)

GPT-2 124M (`d12`), micro batch B=16 per GPU, T=1024, BF16, 50 steps. Numbers are taken from the stable steps at the end of each run.

| GPUs | Allocation | ms/step | tokens/s | MFU | Scaling efficiency |
|---|---|---|---|---|---|
| 1 | shared node | 114.7 | 142,760 | 36.8% | 100% |
| 2 | shared node | 116.9 | 280,200 | 36.0% | 98.1% |
| 4 | shared node | 117.7 | 556,420 | 35.8% | 97.4% |
| 8 (no ZeRO) | full node | 118.3 | 1,106,500 | 35.6% | 96.9% |
| 8 (ZeRO-1) | full node | 116.7 | 1,121,700 | 36.1% | 98.2% |
| 16 (ZeRO-1) | 2 full nodes, InfiniBand | 121.1 | 2,163,392 | 34.7% | 94.7% |

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

- ZeRO-3 does not support writing or resuming from checkpoints yet (it exits with an error if `-n` or `-y 1` is set).
- The `d84` run is unstable over its 5 steps (loss 11 to 25). The same spikes appear identically with ZeRO-2 and ZeRO-3 on `d72`, so this is most likely the learning rate (3e-4, no warmup) at this scale, but it cannot be compared against ZeRO-2 because ZeRO-2 does not fit.

Logs: `experiments/logs/zero2_8gpu*`, `experiments/logs/zero3_8gpu*`, `experiments/logs/zero2_fit_8gpu*`, `experiments/logs/zero3_fit_8gpu*`.

**Longer loss curves** (GPT-2 124M, FineWeb, the full-baseline recipe shortened to 2,000 steps = 1B tokens, 8 GPUs):

| Step | 0 | 500 | 1,000 | 1,500 | 2,000 |
|---|---|---|---|---|---|
| ZeRO-1 val loss | 11.010 | 5.386 | 4.324 | 3.987 | 3.897 |
| ZeRO-2 val loss | 11.010 | 5.382 | 4.325 | 3.981 | 3.892 |
| ZeRO-3 val loss | 11.010 | 5.382 | 4.325 | 3.981 | 3.892 |

ZeRO-2 vs ZeRO-1: the largest val-loss difference is 0.038 at step 700 (right after warmup, where training is least stable); from step 1,000 on it stays within 0.008. ZeRO-3 is bit-identical to ZeRO-2 over all 2,000 steps (train and val loss). Step time with 2 accumulation steps: 523 / 529 / 564 ms. Logs: `experiments/logs/zero_curve_124M*`.

### Where the time goes: Nsight Systems

Rank 0 of `d48` (1.5B), B=2, 8 GPUs, last 4 steps. "Exposed" = NCCL kernel time not overlapped with any compute kernel. Script: [`experiments/profiles/analyze.sh`](experiments/profiles/analyze.sh).

| Per step | Forward compute / exposed NCCL | Backward compute / exposed NCCL | Update: exposed NCCL |
|---|---|---|---|
| ZeRO-1 | 47.9 / 0.0 ms | 109.8 / 2.5 ms | 12.5 ms |
| ZeRO-2 | 48.0 / 0.0 ms | 109.7 / 4.2 ms | 12.6 ms |
| ZeRO-3 | 56.1 / 1.1 ms | 117.0 / 6.3 ms | 0.1 ms |

- The per-layer reduce-scatter in backward is about 90% hidden behind compute for ZeRO-1 and ZeRO-2. The visible communication cost is the all-gather after the AdamW update (12.5 ms, 7% of the step).
- ZeRO-3's per-layer all-gathers are hidden as well (97% in forward, 90% in backward), and it has no all-gather after the update. It is still slower because the **compute itself gets slower**: forward 48.0 → 56.1 ms, backward 109.7 → 117.0 ms. NCCL kernels run on the same GPU at the same time and compete for SMs and memory bandwidth.

### Tensor parallelism

Megatron-style, `-tp N`: ranks are split into tensor-parallel groups of N consecutive ranks (inside a node) and data-parallel groups across them (`ncclCommSplit`).

- `qkv` and `fc` are column-parallel (`qkv` split by attention heads, so attention runs on NH/N local heads with no communication); `attproj` and `fcproj` are row-parallel. Forward all-reduces the output of the two row-parallel layers, backward all-reduces the input gradient of the two column-parallel layers: 4 all-reduces of B×T×C per layer per micro-step, on the critical path.
- LayerNorms, `wte`, `wpe` and the row-parallel biases are replicated. Each rank takes its slice of the full random init, so TP starts from exactly the same weights as a single GPU.
- After the split, all data-parallel and ZeRO code runs over the data-parallel group, so TP composes with ZeRO-1/2/3 without changes there. The gradient norm sums the TP-sliced tensors over the TP group and counts replicated tensors once.

**Correctness** (`d12`, B=8, data-parallel size fixed at 2 so all runs see the same batches, 30 steps):

| Comparison | Step 1 | Max over 30 steps |
|---|---|---|
| TP=2 (4 GPUs) vs TP=1 (2 GPUs) | 0.00006 | 0.022 |
| TP=4 (8 GPUs) vs TP=1 (2 GPUs) | 0.0001 | 0.022 |
| TP=2: ZeRO-1 vs ZeRO-0 | 0 | 0.001 |
| TP=2: ZeRO-2 vs ZeRO-1, ZeRO-3 vs ZeRO-2 | 0 | 0 (bit-identical) |

TP cannot be bit-identical to one GPU: the all-reduce changes the BF16 summation order (6e-5 at step 1). The later drift is of the same size as ZeRO-0 vs ZeRO-1 in upstream. With `-tp 1`, ZeRO-1/2/3 losses are bit-identical to before this change. Logs: `experiments/logs/tp_test_8gpu*`, `experiments/logs/tp_regression_8gpu*`.

### Multi-node: TP inside a node x ZeRO/DP across nodes

2 nodes x 8 A100 (16 GPUs), T=1024. Global batch = B x data-parallel size, in sequences.

| Model | Config | Global batch | Memory / GPU | tokens/s | MFU |
|---|---|---|---|---|---|
| GPT-3 6.7B (`gpt3:c4096`) | TP=8 x DP=2, ZeRO-1, B=8 | 16 | 25.1 GB | **34,851** | 28.5% |
| | ZeRO-3, DP=16, B=1 | 16 | 15.2 GB | 28,786 | 23.5% |
| | TP=8 x DP=2, ZeRO-1, B=16 (TP's max) | 32 | 38.7 GB | 41,442 | 33.9% |
| | ZeRO-3, DP=16, B=4 | 64 | 31.1 GB | **70,833** | **57.9%** |
| GPT-2 12.2B (`d84`) | ZeRO-3, DP=16, B=1 (its max) | 16 | 27.6 GB | 12,970 | 19.5% |
| | TP=4 x DP=4, ZeRO-1, B=2 | 8 | 33.4 GB | 13,128 | 19.7% |
| | TP=4 x DP=4, ZeRO-3, B=4 | 16 | 35.4 GB | **19,149** | **28.8%** |

- At the same global batch, TP is faster than pure ZeRO-3: with ZeRO-3 at B=1 per GPU the matmuls are too small to use the GPU well.
- When memory allows a larger batch, ZeRO-3 wins on 6.7B (57.9% MFU at B=4): its communication overlaps with compute, while TP's 4 all-reduces per layer sit on the critical path.
- On 12.2B, pure ZeRO-3 only fits B=1. TP=4 cuts the per-GPU attention/MLP activations by 4x, so TP x ZeRO-3 fits B=4 and is 48% faster than pure ZeRO-3: the combination beats either alone.
- Plain data parallelism also scales across the two nodes: `d12` on 16 GPUs runs at 121.1 ms/step vs 116.7 ms on 8 (94.7% weak-scaling efficiency vs 1 GPU).

Limitations of TP: no checkpoint writing/loading, random init only (`-e dN` descriptors), heads must divide by the TP size (GPT-2 sizes have 12-36 heads; `gpt3:c4096` has 32), and the LM head (`wte` matmul) is computed redundantly on every TP rank. Text generation (`-s`) is disabled with TP and ZeRO-3, since their forward pass is collective and generation runs on rank 0 only. Logs: `experiments/logs/multinode_16gpu*`.

## Changes to upstream llm.c

- `Makefile`: added a `NCCL_DIR` option so NCCL can be found on systems without `dpkg` (e.g. RHEL-based HPC clusters).
- `llmc/tokenizer.h`: fixed an uninitialized `eot_token` when the tokenizer file is missing, which made generation fail with "Token out of vocabulary". Submitted upstream as a [pull request](https://github.com/karpathy/llm.c/pulls?q=is%3Apr+author%3Ataoyejun238).
- ZeRO-2 and ZeRO-3: new `llmc/zero2.cuh` and `llmc/zero3.cuh`; `llmc/zero.cuh` accepts `-z 2` and `-z 3`; `train_gpt2.cu` changes gradient/parameter allocation, the per-layer pointers in forward and backward, the gradient norm and the AdamW update.
- `train_gpt2.cu`: the all-gather after the AdamW update now waits for the update on the compute stream (event record + wait). Upstream starts it on the NCCL stream without this dependency.
- Tensor parallelism (`-tp`): `llmc/zero.cuh` splits the processes into TP and DP groups and adds `tp_all_reduce`; `train_gpt2.cu` slices parameter and activation shapes, the random init, forward and backward, and the gradient norm. `printf0` and the logger now use the global rank.

## Roadmap

- [x] Build llm.c on the cluster and run baseline scaling (1/2/4/8 GPUs)
- [x] Full baseline: GPT-2 124M trained on 10B tokens of FineWeb, HellaSwag eval (29.8%, above OpenAI's 29.55%)
- [x] Memory ablations on 1.5B (`d48`) with ZeRO-1 and activation recomputation, and a 2.7B (`d60`) fit test
- [x] ZeRO-2 (short-run loss matching, memory and throughput on 1.5B-7.3B)
- [x] ZeRO-3 (bit-identical to ZeRO-2; trains 7.3B at B=2 and 12.2B on one 8-GPU node)
- [x] Longer loss-curve validation of ZeRO-2/3 on GPT-2 124M (2,000 steps, 1B tokens)
- [x] Tensor Parallelism (Megatron-style, composes with ZeRO-1/2/3)
- [x] Multi-node multi-dimensional parallelism (TP x ZeRO/DP on 2 nodes, 6.7B and 12.2B)
- [x] Profiling analysis with Nsight Systems
- [ ] ZeRO-3 / TP checkpoint writing and resuming
- [ ] Overlap the TP all-reduces with compute, vocab-parallel LM head
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

# tensor parallelism inside each node x ZeRO-3 across 2 nodes (16 GPUs)
mpiexec -machinefile $PJM_O_NODEINF -n 16 -npernode 8 ./train_gpt2cu -e d84 -b 4 -t 1024 -x 10 -v 10000 -s 0 \
  -tp 4 -z 3 -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin -j dev/data/tinyshakespeare/tiny_shakespeare_train.bin
~~~

The full baseline is [`experiments/jobs/gpt2_124M_fineweb.sh`](experiments/jobs/gpt2_124M_fineweb.sh) (needs `dev/data/fineweb.py -v 10B` and `dev/data/hellaswag.py`).

The original llm.c README is kept in [`README_llmc.md`](README_llmc.md).
