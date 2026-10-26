/*
ZeRO-3: parameter sharding on top of ZeRO-2.
The 10 large block tensors (qkvw ... fcprojb, param tensors 4..13) are kept only as this rank's 1/N shard,
in the same layout as grads_shard and the AdamW m/v buffers. Right before a layer is used in forward or
backward, its parameters are all-gathered into one of two reusable layer buffers (double-buffered, the next
layer is prefetched on the NCCL stream while the current one computes).
wte, wpe, ln1w, ln1b, lnfw, lnfb stay fully replicated: they are small (except wte), and ln1w/ln1b of
layer l+1 are used inside layer l by the fused residual+layernorm kernel.
Gradients are handled exactly as in ZeRO-2 (zero2.cuh), and the layer buffer layout is shared with it.
*/
#ifndef LLMC_ZERO3_CUH
#define LLMC_ZERO3_CUH

#include "cuda_common.h"
#include "zero.cuh"
#include "zero2.cuh"

typedef struct {
    floatX* param_buf[2];            // full parameters of one layer, double-buffered, same layout as zero2 layer_buf
    floatX* param_shard;             // this rank's parameter shard, same layout as grads_shard
    cudaEvent_t param_ready[2];      // recorded on the NCCL stream when the all-gather into a buffer is done
    cudaEvent_t param_free[2];       // recorded on the compute stream when the layer using a buffer is done
    bool initialized;
} Zero3State;

// Param tensors 4..13 are sharded; 0..3 (wte, wpe, ln1w, ln1b) and 14..15 (lnfw, lnfb) are replicated.
inline bool zero3_is_sharded(int param_tensor_id) { return 4 <= param_tensor_id && param_tensor_id <= 13; }

void zero3_init(Zero3State* z3, const Zero2State* z2, size_t shard_num_parameters) {
    memset(z3, 0, sizeof(Zero3State));
    for (int b = 0; b < 2; b++) {
        cudaCheck(cudaMalloc((void**)&z3->param_buf[b], z2->layer_numel * sizeof(floatX)));
        cudaCheck(cudaEventCreateWithFlags(&z3->param_ready[b], cudaEventDisableTiming));
        cudaCheck(cudaEventCreateWithFlags(&z3->param_free[b], cudaEventDisableTiming));
    }
    cudaCheck(cudaMalloc((void**)&z3->param_shard, shard_num_parameters * sizeof(floatX)));
    z3->initialized = true;
    printf0("ZeRO-3: 2 layer parameter buffers of %zu MiB, parameter shard of %zu MiB\n",
            (z2->layer_numel * sizeof(floatX)) >> 20, (shard_num_parameters * sizeof(floatX)) >> 20);
}

void zero3_free(Zero3State* z3) {
    if (!z3->initialized) { return; }
    for (int b = 0; b < 2; b++) {
        cudaFreeCheck(&z3->param_buf[b]);
        cudaCheck(cudaEventDestroy(z3->param_ready[b]));
        cudaCheck(cudaEventDestroy(z3->param_free[b]));
    }
    cudaFreeCheck(&z3->param_shard);
    z3->initialized = false;
}

// Copy this rank's 1/N slice of every layer of the sharded tensors out of the full parameter buffer.
// full_params uses the standard layout (16 tensors back to back, block tensors layer-major).
void zero3_shard_params(Zero3State* z3, const Zero2State* z2, const floatX* full_params,
                        const size_t* param_elements, int num_layers, int rank, int nproc) {
    size_t full_offset = 0;
    for (int i = 0; i < ZERO2_NUM_PARAM_TENSORS; i++) {
        if (zero3_is_sharded(i)) {
            size_t layer_size = z2->layer_sizes[i - 2];
            size_t n = layer_size / nproc;
            // one row per layer: copy n elements starting at this rank's offset within the layer
            cudaCheck(cudaMemcpy2D(z3->param_shard + z2->shard_offsets[i], n * sizeof(floatX),
                                   full_params + full_offset + rank * n, layer_size * sizeof(floatX),
                                   n * sizeof(floatX), num_layers, cudaMemcpyDeviceToDevice));
        }
        full_offset += param_elements[i];
    }
}

// Enqueue the all-gather of one layer's sharded parameters into param_buf[buf] on the NCCL stream.
// Waits until the compute stream has finished with the layer that last used this buffer.
void zero3_gather_layer(Zero3State* z3, const Zero2State* z2, int layer, int buf, MultiGpuConfig* config) {
#ifdef MULTI_GPU
    const int nproc = config->num_processes;
    cudaCheck(cudaStreamWaitEvent(config->nccl_stream, z3->param_free[buf], 0));
    ncclCheck(ncclGroupStart());
    for (int j = 2; j < ZERO2_NUM_BLOCK_TENSORS; j++) {  // block tensors 0,1 (ln1w, ln1b) are replicated
        size_t n = z2->layer_sizes[j] / nproc;
        ncclCheck(ncclAllGather(z3->param_shard + z2->shard_offsets[j + 2] + (size_t)layer * n,
                                z3->param_buf[buf] + z2->layer_offsets[j],
                                n, ncclFloatX, config->nccl_comm, config->nccl_stream));
    }
    ncclCheck(ncclGroupEnd());
    cudaCheck(cudaEventRecord(z3->param_ready[buf], config->nccl_stream));
#endif
}

// Before a layer runs: the compute stream waits for its all-gather. Returns the 12 block parameter
// pointers in ParameterTensors order; entries 0,1 (ln1w, ln1b) are NULL, use the replicated tensors.
void zero3_acquire_layer_params(Zero3State* z3, const Zero2State* z2, int buf, cudaStream_t compute_stream,
                                floatX* out[ZERO2_NUM_BLOCK_TENSORS]) {
    cudaCheck(cudaStreamWaitEvent(compute_stream, z3->param_ready[buf], 0));
    for (int j = 0; j < ZERO2_NUM_BLOCK_TENSORS; j++) {
        out[j] = (j < 2) ? NULL : z3->param_buf[buf] + z2->layer_offsets[j];
    }
}

// After a layer's kernels are enqueued: mark the buffer reusable once they have run.
void zero3_release_layer_params(Zero3State* z3, int buf, cudaStream_t compute_stream) {
    cudaCheck(cudaEventRecord(z3->param_free[buf], compute_stream));
}

#endif // LLMC_ZERO3_CUH
