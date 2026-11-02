/*
Context parallelism (CP) for the non-cuDNN attention in attention.cuh.

Every sequence of T tokens is split into cp contiguous chunks of Tl = T / cp tokens, and CP rank r runs
the whole model on chunk r only (tokens [r * Tl, (r + 1) * Tl) of every sequence). Everything except
attention works token by token, so it runs unchanged on the local chunk. Attention needs, for each local
query, the keys and values of all earlier tokens of its sequence, which live on CP ranks 0 .. r:

forward:  all-gather K and V over the CP group, then attention of the Tl local queries against the
          Tk = (r + 1) * Tl keys they can see (causal mask with the query offset r * Tl).
backward: all-gather K and V again (not stored, to save memory), compute dQ for the local queries and
          partial dK, dV for all Tk keys, then reduce-scatter dK, dV so that every rank gets the summed
          gradients of its own keys and values.

This is the all-gather flavour of CP (as in Llama 3); Megatron's ring attention passes the K/V chunks
around a ring instead, to overlap the transfers with the attention compute.

Layouts (per batch element b and local head h, HS = head size):
  qkvr       (3, B, NH, Tl, HS)      this rank's Q, K, V (as in attention.cuh, with T = Tl)
  gathered   (2, cp, B, NH, Tl, HS)  K and V of all CP ranks, as all-gathered (rank-major)
  full       (2, B, NH, Tk, HS)      K and V of the first Tk tokens of the sequence
  att        (B, NH, Tl, Tk)         attention probabilities of the local queries
*/
#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"
// attention.cuh (permute kernels) and zero.cuh (multi_gpu_config) are included before this file

// ----------------------------------------------------------------------------
// CUDA kernels

// gathered (2, cp, B, NH, Tl, HS) -> full (2, B, NH, Tk, HS): token t of the sequence is token t % Tl of rank t / Tl
__global__ void cp_gathered_to_full_kernel(floatX* full, const floatX* gathered, int B, int NH, int Tl, int Tk, int HS, int cp) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t part = (size_t)B * NH * Tk * HS;
    if (idx >= 2 * part) { return; }
    int p = idx / part;           // 0: K, 1: V
    size_t rest = idx % part;
    int d = rest % HS;   rest /= HS;
    int t = rest % Tk;   rest /= Tk;
    int h = rest % NH;
    int b = rest / NH;
    int r = t / Tl;
    int tl = t % Tl;
    size_t src = (size_t)p * cp * B * NH * Tl * HS + ((((size_t)r * B + b) * NH + h) * Tl + tl) * HS + d;
    full[idx] = gathered[src];
}

// the reverse, for the gradients: full (2, B, NH, Tk, HS) -> gathered (2, cp, B, NH, Tl, HS), with zeros for the
// tokens t >= Tk that this rank's queries cannot see
__global__ void cp_full_to_gathered_kernel(floatX* gathered, const floatX* full, int B, int NH, int Tl, int Tk, int HS, int cp) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t part = (size_t)cp * B * NH * Tl * HS;
    if (idx >= 2 * part) { return; }
    int p = idx / part;
    size_t rest = idx % part;
    int d = rest % HS;   rest /= HS;
    int tl = rest % Tl;  rest /= Tl;
    int h = rest % NH;   rest /= NH;
    int b = rest % B;
    int r = rest / B;
    int t = r * Tl + tl;
    floatX val = (floatX)0.0f;
    if (t < Tk) {
        val = full[(size_t)p * B * NH * Tk * HS + (((size_t)b * NH + h) * Tk + t) * HS + d];
    }
    gathered[idx] = val;
}

// softmax_forward_kernel5 from attention.cuh for a (Tq, Tk) block of the attention matrix whose first
// query sits at position q_offset of the sequence: query i sees keys 0 .. q_offset + i.
// inp, out shape: (N, Tq, Tk), where N = B * NH
__global__ void cp_softmax_forward_kernel(floatX* out, float inv_temperature, const floatX* inp, int N, int Tq, int Tk, int q_offset) {
    assert(Tk % 4  == 0);
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int idx = (gridDim.x - blockIdx.x - 1) * num_warps + warp_id; // backward order, as in kernel5
    if(idx >= N * Tq) {
        return;
    }
    int own_pos = q_offset + idx % Tq;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (Tk,)
    const floatX* x = inp + (size_t)idx * Tk;

    const float flt_max = 340282346638528859811704183484516925440.0f;
    float maxval = -flt_max;
    float sumval = 0.0f;

    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, regarray[k]);
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (regarray[k] - maxval));
        }
    }

    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, (float)x[4*pos_by_4 + lane_id]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * ((float)x[4*pos_by_4 + lane_id] - maxval));
    }

    float global_maxval = warpReduceMax(maxval);
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = warpReduceSum(sumval);
    float norm = 1.f / sum;

    for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
        float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + (size_t)idx * Tk + i, (floatX)(ev * norm));
    }
}

// softmax_autoregressive_backward_inplace_kernel from attention.cuh for a (Tq, Tk) block with query offset
__global__ void cp_softmax_backward_inplace_kernel(floatX* datt, const floatX* att, int Tq, int Tk, int q_offset, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;

    int t0 = Tq - 1 - T_per_block*blockIdx.x;
    int idx = blockIdx.y;

    att += (size_t)idx * Tq * Tk;
    datt += (size_t)idx * Tq * Tk;

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        int own_pos = q_offset + t;
        const floatX* att_bth = att + (size_t)t * Tk;
        const floatX* datt_bth = datt + (size_t)t * Tk;
        floatX* dpreatt_bth = datt + (size_t)t * Tk;

        float local_sum = 0;
        for (int t2 = threadIdx.x; t2 <= own_pos; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }

        local_sum = blockReduce<warpReduceSum>(local_sum);

        for (int t3 = threadIdx.x; t3 < Tk; t3 += BlockSize) {
            if(t3 <= own_pos) {
                float acc = (float) __ldcs(att_bth + t3) * ((float) __ldcs(datt_bth + t3) - local_sum);
                __stcs(dpreatt_bth + t3, (floatX) (scale * acc));
            } else {
                __stcs(dpreatt_bth + t3, (floatX)0.f);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// All-gather this rank's K and V (from qkvr) over the CP group and lay them out as full (2, B, NH, Tk, HS).
// `gathered` is scratch of 2 * cp * B * NH * Tl * HS elements.
void cp_gather_kv(floatX* full, floatX* gathered, const floatX* qkvr, int B, int Tl, int C, int NH, cudaStream_t stream) {
    const int HS = C / NH;
    const int cp = multi_gpu_config.cp_size;
    const int Tk = (multi_gpu_config.cp_rank + 1) * Tl;
    const size_t chunk = (size_t)B * Tl * C;  // one of Q, K, V of one rank
#ifdef MULTI_GPU
    ncclCheck(ncclGroupStart());
    ncclCheck(ncclAllGather(qkvr + 1 * chunk, gathered, chunk, ncclFloatX, multi_gpu_config.cp_comm, stream));
    ncclCheck(ncclAllGather(qkvr + 2 * chunk, gathered + cp * chunk, chunk, ncclFloatX, multi_gpu_config.cp_comm, stream));
    ncclCheck(ncclGroupEnd());
#endif
    const int block_size = 256;
    size_t n = 2 * (size_t)B * NH * Tk * HS;
    cp_gathered_to_full_kernel<<<CEIL_DIV(n, block_size), block_size, 0, stream>>>(full, gathered, B, NH, Tl, Tk, HS, cp);
    cudaCheck(cudaGetLastError());
}

// Same interface as attention_forward, for the Tl local tokens (T = Tl), plus a CP buffer `kv` of
// 4 * B * (cp * Tl) * C elements. att is (B, NH, Tl, Tk).
void attention_forward_cp(floatX* out, floatX* qkvr, floatX* att, floatX* inp, floatX* kv,
                          int B, int Tl, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int HS = C / NH;
    const int cp = multi_gpu_config.cp_size;
    const int q_offset = multi_gpu_config.cp_rank * Tl;
    const int Tk = q_offset + Tl;  // keys this rank's queries can see
    assert(Tl % 4 == 0);

    // local Q, K, V as in attention_forward (stored for backward)
    floatX *q, *k, *v;
    q = qkvr + 0 * B * Tl * C;
    k = qkvr + 1 * B * Tl * C;
    v = qkvr + 2 * B * Tl * C;
    int total_threads = B * NH * Tl * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, Tl, NH, HS);

    // K, V of the first Tk tokens of every sequence
    floatX* gathered = kv;
    floatX* full = kv + 2 * (size_t)B * cp * Tl * C;
    cp_gather_kv(full, gathered, qkvr, B, Tl, C, NH, stream);
    const floatX* k_full = full;
    const floatX* v_full = full + (size_t)B * NH * Tk * HS;

    // preatt (B, NH, Tl, Tk) = q @ k_full^T
    floatX* preatt = inp;
    matmul_cublaslt(preatt, k_full, q, nullptr, Tk, Tl, HS, stream, true, false, B * NH, Tk * HS, Tl * HS, Tl * Tk);

    float scale = 1.f / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * Tl * WARP_SIZE, block_size);
    cp_softmax_forward_kernel<<<grid_size, block_size, 0, stream>>>(att, scale, preatt, B * NH, Tl, Tk, q_offset);

    // y (B, NH, Tl, HS) = att @ v_full
    floatX* vaccum = inp;
    matmul_cublaslt(vaccum, v_full, att, nullptr, HS, Tl, Tk, stream, false, false, B * NH, Tk * HS, Tl * Tk, Tl * HS);

    num_blocks = CEIL_DIV(B * Tl * C, block_size);
    unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, Tl, NH, HS);
    cudaCheck(cudaGetLastError());
}

// Same interface as attention_backward, plus the CP buffer `kv` (see attention_forward_cp).
void attention_backward_cp(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                           const floatX* dout,
                           const floatX* qkvr, const floatX* att, floatX* kv,
                           int B, int Tl, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int HS = C / NH;
    const int cp = multi_gpu_config.cp_size;
    const int q_offset = multi_gpu_config.cp_rank * Tl;
    const int Tk = q_offset + Tl;
    const size_t chunk = (size_t)B * Tl * C;

    const floatX* q = qkvr;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * chunk;
    dk = dqkvr + 1 * chunk;
    dv = dqkvr + 2 * chunk;

    // two halves of kv: A holds the all-gathered K, V and later the full dK, dV;
    // F holds the full K, V and later the gathered-layout dK, dV for the reduce-scatter
    floatX* bufA = kv;
    floatX* bufF = kv + 2 * (size_t)B * cp * Tl * C;
    cp_gather_kv(bufF, bufA, qkvr, B, Tl, C, NH, stream);
    const floatX* k_full = bufF;
    const floatX* v_full = bufF + (size_t)B * NH * Tk * HS;
    floatX* dk_full = bufA;
    floatX* dv_full = bufA + (size_t)B * NH * Tk * HS;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * Tl * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(scratch, dout, B, Tl, NH, HS);
    // datt (B, NH, Tl, Tk) = dy @ v_full^T
    matmul_cublaslt(datt, v_full, scratch, nullptr, Tk, Tl, HS, stream, true, false, B * NH, Tk * HS, Tl * HS, Tl * Tk);
    // dv_full (B, NH, Tk, HS) = att^T @ dy
    matmul_cublaslt(dv_full, scratch, att, nullptr, HS, Tk, Tl, stream, false, true, B * NH, Tl * HS, Tl * Tk, Tk * HS);
    const float scale = 1.0f / sqrtf((float)HS);
    cp_softmax_backward_inplace_kernel<<<dim3(Tl / 4, B * NH), 256, 0, stream>>>(datt, att, Tl, Tk, q_offset, scale);
    const floatX* dpreatt = datt;
    // dq (B, NH, Tl, HS) = dpreatt @ k_full
    matmul_cublaslt(dq, k_full, dpreatt, nullptr, HS, Tl, Tk, stream, false, false, B * NH, Tk * HS, Tl * Tk, Tl * HS);
    // dk_full (B, NH, Tk, HS) = dpreatt^T @ q
    matmul_cublaslt(dk_full, q, dpreatt, nullptr, HS, Tk, Tl, stream, false, true, B * NH, Tl * HS, Tl * Tk, Tk * HS);

    // every rank holds partial dK, dV of the keys of ranks 0 .. cp_rank: sum them on their owners
    size_t n = 2 * (size_t)cp * chunk;
    cp_full_to_gathered_kernel<<<CEIL_DIV(n, block_size), block_size, 0, stream>>>(bufF, bufA, B, NH, Tl, Tk, HS, cp);
    cudaCheck(cudaGetLastError());
#ifdef MULTI_GPU
    ncclCheck(ncclGroupStart());
    ncclCheck(ncclReduceScatter(bufF, dk, chunk, ncclFloatX, ncclSum, multi_gpu_config.cp_comm, stream));
    ncclCheck(ncclReduceScatter(bufF + cp * chunk, dv, chunk, ncclFloatX, ncclSum, multi_gpu_config.cp_comm, stream));
    ncclCheck(ncclGroupEnd());
#endif

    // backward into inp
    num_blocks = CEIL_DIV(B * NH * Tl * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(dinp, dq, dk, dv, B, Tl, NH, HS);
    cudaCheck(cudaGetLastError());
}
