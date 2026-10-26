/*
ZeRO-2: optimizer state + gradient sharding.
Block gradients (param tensors 2..13): computed into one of two reusable layer buffers,
reduce-scattered right after the layer's backward, accumulated into grads_shard.
Non-block gradients (wte, wpe, lnfw, lnfb): full-size buffer, reduce-scattered on the last micro-step.
grads_shard has the same layout as the sharded AdamW m/v buffers.

ZeRO-2：优化器状态和梯度都切分，每张卡只保存自己负责的那 1/N。
层内张量（编号 2..13）的梯度：写进两个轮流复用的层缓冲区之一，
这一层反向一结束就做 reduce-scatter，本卡那 1/N 累加到梯度分片 grads_shard。
层外张量（wte、wpe、lnfw、lnfb）的梯度：保留完整缓冲区，只在最后一个 micro-step 做 reduce-scatter。
grads_shard 的布局和切分后的 AdamW m、v 完全一样，所以优化器几乎不用改。
*/

// Include guard: skip the whole file if it has already been included.
// 头文件保护：如果这个文件已经被 include 过，就跳过整个文件，避免重复定义。
#ifndef LLMC_ZERO2_CUH
#define LLMC_ZERO2_CUH

#include <string.h>        // memset
#include "cuda_common.h"   // floatX, cudaCheck, cudaFreeCheck
#include "zero.cuh"        // MultiGpuConfig, ncclCheck, ncclFloatX, printf0

constexpr int ZERO2_NUM_PARAM_TENSORS = 16;  // must equal NUM_PARAMETER_TENSORS
constexpr int ZERO2_NUM_BLOCK_TENSORS = 12;  // param tensors 2..13 (ln1w ... fcprojb)
// ZeRO-2 state: all buffers, events and offsets, filled in by zero2_init.
// ZeRO-2 的状态：所有缓冲区、event 和偏移量都打包在这里，由 zero2_init 填好。
typedef struct {
    // Two layer buffers ("boxes"), each holding the full gradients of one layer.
    // Layer l uses layer_buf[l % 2], so one layer can be reduced while the next is computed.
    // 两个层缓冲区（"盒子"），每个装一层的完整梯度（本卡自己算的，还没平均）。
    // 第 l 层用 layer_buf[l % 2]：一个在通信时，另一个可以继续被计算写入（双缓冲）。
    floatX* layer_buf[2];

    // Reduce-scatter output of one layer (1/N of a layer). NCCL overwrites its output,
    // but gradients must accumulate across micro-steps, so we land here first, then add.
    // 一层 reduce-scatter 的输出（一层的 1/N）。NCCL 只会覆盖写入、不会累加，
    // 而梯度要跨 micro-step 累加，所以先写到这里，再用 kernel 加进 grads_shard。
    floatX* layer_shard_tmp;

    // This rank's accumulated, averaged gradient shard (1/N of all parameters).
    // The optimizer reads gradients from here.
    // 本卡负责的那 1/N 平均梯度（全部参数的 1/N），跨层、跨 micro-step 累加。
    // 优化器更新参数时从这里读梯度。
    floatX* grads_shard;

    // One event per layer buffer, recorded on the NCCL stream after the buffer has been read.
    // The compute stream waits on it before overwriting the buffer.
    // 每个层缓冲区一个 event：通信流读完这个缓冲区后记录它，
    // 计算流在覆盖这个缓冲区之前先等它，避免把还没传完的梯度覆盖掉。
    cudaEvent_t layer_buf_free[2];

    // Number of parameters in one transformer layer (size of one layer buffer).
    // 一个 Transformer 层的参数个数，也就是一个层缓冲区的大小（元素个数）。
    size_t layer_numel;

    // Start of each of the 12 block tensors inside a layer buffer (in elements).
    // 12 个层内张量各自在层缓冲区里的起始位置（单位：元素个数）。
    size_t layer_offsets[ZERO2_NUM_BLOCK_TENSORS];

    // Per-layer size of each of the 12 block tensors (in elements).
    // 12 个层内张量每一层各有多少个元素。
    size_t layer_sizes[ZERO2_NUM_BLOCK_TENSORS];

    // Start of each of the 16 parameter tensors inside grads_shard (in elements).
    // 16 个参数张量各自在 grads_shard 里的起始位置（单位：元素个数）。
    size_t shard_offsets[ZERO2_NUM_PARAM_TENSORS];

    // Set by zero2_init; zero2_free does nothing if it is false.
    // zero2_init 结束时设为 true；zero2_free 看到 false 就什么都不做。
    bool initialized;
} Zero2State;

// Called once before training: compute the two memory layouts and allocate all ZeRO-2 buffers.
// 训练开始前调用一次：算好两套内存布局，申请 ZeRO-2 要用的全部显存和 event。
//   z                    : state to fill in (model->zero2)           要填好的状态（就是 model->zero2）
//   param_elements       : element count of each of the 16 tensors   16 个参数张量各自的元素个数
//                          (block tensors: all L layers combined)    （层内张量是 L 层加在一起的）
//   num_layers           : number of transformer layers L            层数 L
//   shard_num_parameters : parameters owned by this rank (total / N) 本卡负责的参数个数（总数 / N）
//   nproc                : number of GPUs N                          GPU 数 N
void zero2_init(Zero2State* z, const size_t* param_elements, int num_layers,
                size_t shard_num_parameters, int nproc) {
    // Zero the struct: pointers become NULL, layer_numel starts at 0, initialized = false.
    // 结构体清零：指针都是 NULL，layer_numel 从 0 开始累加，initialized = false。
    memset(z, 0, sizeof(Zero2State));

    // Layout 1: where each tensor starts in grads_shard.
    // Each tensor takes 1/N of its size, in tensor order, so its start is
    // (sum of the full sizes of all previous tensors) / N. Same layout as the sharded AdamW m/v.
    // 布局 1：每个张量在 grads_shard 里从哪里开始。
    // 每个张量在 grads_shard 里占原来的 1/N，按编号 0..15 依次排，
    // 所以起点 = (前面所有张量的完整大小之和) / N。和切分后的 AdamW m、v 布局相同。
    size_t full_offset = 0;  // sum of full sizes of previous tensors / 前面所有张量的完整大小之和
    for (int i = 0; i < ZERO2_NUM_PARAM_TENSORS; i++) {
        z->shard_offsets[i] = full_offset / nproc;  // start of tensor i in grads_shard / 张量 i 在 grads_shard 里的起点
        full_offset += param_elements[i];           // accumulate for the next tensor / 累加，给下一个张量用
    }

    // Layout 2: how the 12 block tensors of one layer are packed into a layer buffer (back to back).
    // 布局 2：一层的 12 个层内张量在层缓冲区里怎么放（首尾相接）。
    for (int j = 0; j < ZERO2_NUM_BLOCK_TENSORS; j++) {
        // Per-layer size of block tensor j (global tensor id j + 2).
        // 层内张量 j（全局编号 j + 2）每一层的大小：L 层的总大小除以 L。
        z->layer_sizes[j] = param_elements[j + 2] / num_layers;
        // Its start inside the buffer = sum of the sizes of the previous tensors.
        // 它在缓冲区里的起点 = 前面所有张量的大小之和。
        z->layer_offsets[j] = z->layer_numel;
        // Accumulate; after the loop this is the size of one whole layer.
        // 累加；循环结束后就是一整层的大小。
        z->layer_numel += z->layer_sizes[j];
        // Reduce-scatter splits every tensor into N equal parts, so it must divide evenly.
        // Fail at startup instead of producing wrong results later.
        // reduce-scatter 要把每个张量均分成 N 份，必须能整除。
        // 启动时就报错退出，而不是等到训练中途出现奇怪的错误。
        if (z->layer_sizes[j] % nproc != 0) {
            fprintf(stderr, "ZeRO-2: tensor %d not divisible by %d processes\n", j + 2, nproc);
            exit(EXIT_FAILURE);
        }
    }

    // Allocate everything once here; during training the buffers are only reused, never reallocated.
    // 所有显存都在这里一次申请好；训练过程中只复用，不再申请（cudaMalloc 很慢，而且会同步设备）。
    for (int b = 0; b < 2; b++) {
        // Layer buffer b: holds the full gradients of one layer.
        // 层缓冲区 b：装一层的完整梯度。
        cudaCheck(cudaMalloc((void**)&z->layer_buf[b], z->layer_numel * sizeof(floatX)));
        // Event for layer buffer b; used only for synchronization, so timing is disabled.
        // 层缓冲区 b 的 event；只用来同步、不用来计时，所以关掉计时，开销更小。
        cudaCheck(cudaEventCreateWithFlags(&z->layer_buf_free[b], cudaEventDisableTiming));
    }
    // Reduce-scatter output: 1/N of one layer.
    // reduce-scatter 的输出：一层的 1/N。
    cudaCheck(cudaMalloc((void**)&z->layer_shard_tmp, (z->layer_numel / nproc) * sizeof(floatX)));
    // This rank's gradient shard: 1/N of all parameters.
    // 本卡的梯度分片：全部参数的 1/N。
    cudaCheck(cudaMalloc((void**)&z->grads_shard, shard_num_parameters * sizeof(floatX)));

    // Mark as initialized so zero2_free knows there is something to release.
    // 标记已初始化，zero2_free 据此知道有东西要释放。
    z->initialized = true;

    // Print the buffer sizes on rank 0 only (>> 20 converts bytes to MiB), to confirm ZeRO-2 is active.
    // 只在 0 号卡打印缓冲区大小（>> 20 是把字节换成 MiB），用来确认 ZeRO-2 生效了。
    printf0("ZeRO-2: 2 layer buffers of %zu MiB, gradient shard of %zu MiB\n",
            (z->layer_numel * sizeof(floatX)) >> 20, (shard_num_parameters * sizeof(floatX)) >> 20);
}


// Called once after training: release everything zero2_init allocated.
// Safe to call even if zero2_init was never called (e.g. running without ZeRO-2).
// 训练结束时调用一次：释放 zero2_init 申请的全部显存和 event。
// 即使没调用过 zero2_init（比如没开 ZeRO-2）也可以放心调用。
void zero2_free(Zero2State* z) {
    // Nothing was allocated (or already freed): do nothing.
    // 没有申请过（或者已经释放过）：什么都不做，避免重复释放。
    if (!z->initialized) { return; }
    for (int b = 0; b < 2; b++) {
        // Free layer buffer b (cudaFreeCheck also resets the pointer to NULL).
        // 释放层缓冲区 b（cudaFreeCheck 释放后还会把指针设成 NULL）。
        cudaFreeCheck(&z->layer_buf[b]);
        // Destroy the event of layer buffer b.
        // 销毁层缓冲区 b 的 event。
        cudaCheck(cudaEventDestroy(z->layer_buf_free[b]));
    }
    // Free the transfer buffer and the gradient shard.
    // 释放中转区和梯度分片。
    cudaFreeCheck(&z->layer_shard_tmp);
    cudaFreeCheck(&z->grads_shard);
    // Mark as not initialized, so a second call does nothing.
    // 标记为未初始化，再调用一次也不会出错。
    z->initialized = false;
}

// Before a layer's backward: wait until the last reduce-scatter that read this buffer
// is done, then zero it (all backward kernels do += into parameter gradients).
// 每一层反向开始之前调用：把层缓冲区 buf 准备成"可以安全写入、而且是干净的"状态。
//   z              : ZeRO-2 state                          ZeRO-2 的状态
//   buf            : which layer buffer, 0 or 1 (l % 2)    用哪个层缓冲区，0 或 1（调用时传 l % 2）
//   compute_stream : the compute stream (main_stream)      计算流（就是 main_stream）
void zero2_acquire_layer_buf(Zero2State* z, int buf, cudaStream_t compute_stream) {
    // Compute stream waits until the communication stream has finished reading this buffer
    // (it may still hold layer l+2). Never-recorded events pass immediately.
    // 计算流等通信流把这个层缓冲区读完（里面可能还是第 l+2 层的梯度，正在做 reduce-scatter）。
    // 第一次使用时 event 还没被记录过，会直接通过，不会卡住。
    cudaCheck(cudaStreamWaitEvent(compute_stream, z->layer_buf_free[buf], 0));
    // Zero the whole buffer on the compute stream: backward kernels accumulate with +=,
    // so leftovers from the previous layer would be mixed in otherwise.
    // 在计算流上把整个层缓冲区清零：反向 kernel 是用 += 写梯度的，不清零会把上一次的数字混进来。
    cudaCheck(cudaMemsetAsync(z->layer_buf[buf], 0, z->layer_numel * sizeof(floatX), compute_stream));
}


// The 12 block-gradient pointers of one layer, in ParameterTensors order.
// 给出层缓冲区 buf 里 12 个层内张量梯度各自的起始地址，按 ParameterTensors 的顺序
// （ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw, fcprojb）。
// 反向 kernel 会把梯度写到这些地址，也就是写进层缓冲区。
//   z   : ZeRO-2 state (read only)              ZeRO-2 的状态（只读）
//   buf : which layer buffer, 0 or 1 (l % 2)    用哪个层缓冲区，0 或 1（调用时传 l % 2）
//   out : output, 12 pointers                   输出：12 个指针
void zero2_layer_grad_ptrs(const Zero2State* z, int buf, floatX* out[ZERO2_NUM_BLOCK_TENSORS]) {
    for (int j = 0; j < ZERO2_NUM_BLOCK_TENSORS; j++) {
        // Start of the buffer + offset of tensor j inside it.
        // 层缓冲区的开头 + 第 j 个层内张量在里面的偏移量。
        out[j] = z->layer_buf[buf] + z->layer_offsets[j];
    }
}
__global__ void zero2_accumulate_kernel(floatX* dst, const floatX* src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dst[i] = (floatX)((float)dst[i] + (float)src[i]); }
}
// After a layer's backward, every micro-step: reduce-scatter, then accumulate into grads_shard.
// 每个 micro-step 里，每个 Transformer 层的反向算完之后调用一次：
// 把这一层的梯度在各卡之间求平均，本卡那 1/N 累加进梯度分片，然后宣布层缓冲区可以复用。
//   z              : ZeRO-2 state                                         ZeRO-2 的状态
//   buf            : layer buffer holding this layer's gradients (l % 2)  这一层的梯度在哪个层缓冲区（l % 2）
//   layer          : layer index l, picks the position in grads_shard     层号 l，用来确定在梯度分片里的位置
//   config         : multi-GPU config (GPU count, NCCL comm, NCCL stream) 多卡配置（卡数、NCCL 通信器、通信流）
//   compute_stream : the compute stream (main_stream)                     计算流（就是 main_stream）
void zero2_reduce_layer(Zero2State* z, int buf, int layer, MultiGpuConfig* config, cudaStream_t compute_stream) {
#ifdef MULTI_GPU  // only compiled for multi-GPU builds / 只有编译多卡版本时才有这段代码
    // Number of GPUs N.
    // GPU 数量 N。
    const int nproc = config->num_processes;

    // Step 1: the communication stream waits until the compute stream has finished this layer,
    // so reduce-scatter reads complete gradients.
    // 第 1 步：通信流等计算流把这一层算完，保证 reduce-scatter 读到的是完整的梯度。
    cudaCheck(cudaEventRecord(config->compute_nccl_sync, compute_stream));            // compute stream: record "layer done" / 计算流：记录"这一层算完了"
    cudaCheck(cudaStreamWaitEvent(config->nccl_stream, config->compute_nccl_sync, 0)); // NCCL stream: wait for it / 通信流：等这个记号

    // Step 2: reduce-scatter each of the 12 tensors from the layer buffer into layer_shard_tmp.
    // Each tensor is split into N parts separately; this rank receives the average of its part.
    // The group merges the 12 calls into one NCCL launch.
    // 第 2 步：对 12 个层内张量各做一次 reduce-scatter，从层缓冲区写进中转区。
    // 每个张量单独切成 N 份，各卡求平均，本卡只拿自己那一份。
    // GroupStart/GroupEnd 把 12 次调用合成一组，NCCL 一次性执行。
    ncclCheck(ncclGroupStart());
    for (int j = 0; j < ZERO2_NUM_BLOCK_TENSORS; j++) {
        ncclCheck(ncclReduceScatter(z->layer_buf[buf] + z->layer_offsets[j],         // input: tensor j in the layer buffer / 输入：层缓冲区里第 j 个张量
                                    z->layer_shard_tmp + z->layer_offsets[j] / nproc, // output: its slot in layer_shard_tmp / 输出：中转区里对应的位置
                                    z->layer_sizes[j] / nproc, ncclFloatX, ncclAvg,   // elements per rank, bf16, average / 每张卡分到的元素个数，bf16，求平均
                                    config->nccl_comm, config->nccl_stream));         // on the NCCL stream / 在通信流上执行
    }
    ncclCheck(ncclGroupEnd());

    // Step 3: add each tensor's slice from layer_shard_tmp into grads_shard (accumulates across micro-steps).
    // Runs on the NCCL stream, so it always starts after the reduce-scatter above.
    // 第 3 步：把中转区里每个张量的那一份，加进梯度分片里第 layer 层对应的位置（跨 micro-step 累加）。
    // 放在通信流上，所以一定在上面的 reduce-scatter 做完之后才执行。
    for (int j = 0; j < ZERO2_NUM_BLOCK_TENSORS; j++) {
        // Elements of tensor j owned by this rank in one layer.
        // 第 j 个层内张量在一层里本卡分到的元素个数。
        size_t n = z->layer_sizes[j] / nproc;
        // Destination: start of tensor j in grads_shard, then skip the previous `layer` layers.
        // 目标：第 j 个张量在梯度分片里的起点，再跳过前面 layer 层。
        floatX* dst = z->grads_shard + z->shard_offsets[j + 2] + (size_t)layer * n;
        // Source: slot of tensor j in layer_shard_tmp.
        // 来源：中转区里第 j 个张量的位置。
        const floatX* src = z->layer_shard_tmp + z->layer_offsets[j] / nproc;
        // dst += src, 512 threads per block, enough blocks to cover n elements.
        // dst += src，每个 block 512 个线程，block 数向上取整，正好覆盖 n 个元素。
        zero2_accumulate_kernel<<<CEIL_DIV(n, 512), 512, 0, config->nccl_stream>>>(dst, src, n);
        // Check that the kernel launched correctly.
        // 检查 kernel 启动有没有出错。
        cudaCheck(cudaGetLastError());
    }

    // Step 4: mark this layer buffer as free. The compute stream waits on this event
    // (in zero2_acquire_layer_buf) before writing layer l-2 into the same buffer.
    // 第 4 步：在通信流上记录"层缓冲区 buf 读完了"。
    // 以后第 l-2 层要用同一个层缓冲区时，计算流会在 zero2_acquire_layer_buf 里先等这个 event。
    cudaCheck(cudaEventRecord(z->layer_buf_free[buf], config->nccl_stream));
#endif
}

// Last micro-step, non-block tensors: reduce-scatter straight into grads_shard.
template<int N>
void zero2_reduce_full_grads(floatX* const (&full)[N], const int (&tensor_ids)[N], const size_t (&sizes)[N],
                             Zero2State* z, MultiGpuConfig* config, cudaStream_t compute_stream) {
#ifdef MULTI_GPU
    const int nproc = config->num_processes;
    cudaCheck(cudaEventRecord(config->compute_nccl_sync, compute_stream));
    cudaCheck(cudaStreamWaitEvent(config->nccl_stream, config->compute_nccl_sync, 0));
    ncclCheck(ncclGroupStart());
    for (int i = 0; i < N; i++) {
        assert(sizes[i] % nproc == 0);
        ncclCheck(ncclReduceScatter(full[i], z->grads_shard + z->shard_offsets[tensor_ids[i]],
                                    sizes[i] / nproc, ncclFloatX, ncclAvg,
                                    config->nccl_comm, config->nccl_stream));
    }
    ncclCheck(ncclGroupEnd());
#endif
}

#endif // LLMC_ZERO2_CUH