#include <cstdint>
#include <float.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "thriftattention/sm80/cuda_common.cuh"

template <typename T>
__device__ inline T int8_attention_from_float(float value);

template <>
__device__ inline half int8_attention_from_float<half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 int8_attention_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T, bool CAUSAL, int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int INT8_HEAD_DIM, int SCALE_DIM, int NUM_WARPS, int WARP_Q>
__global__ void int8_attention_kernel(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    const float* S_Q,
    const float* S_K,
    const float* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads
) {
    if (threadIdx.x != 0) {
        return;
    }

    const int q_idx = blockIdx.x;
    const int q_token = q_idx % q_len;
    const int q_head = (q_idx / q_len) % num_q_heads;
    const int batch = q_idx / (q_len * num_q_heads);
    const int kv_head = q_head * num_kv_heads / num_q_heads;

    const int q_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * HEAD_DIM;
    const int o_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * HEAD_DIM;
    const int sq_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * SCALE_DIM;

    // Phase 1: find the largest QK score for numerical stability.
    float max_score = -INFINITY;
    for (int kv_token = 0; kv_token < kv_len; kv_token++) {
        const int k_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * HEAD_DIM;
        const int sk_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * SCALE_DIM;

        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            const int group = d / 32;
            const int q_val = int(Q[q_offset + d]);
            const int k_val = int(K[k_offset + d]);
            const float scale = S_Q[sq_offset + group] * S_K[sk_offset + group];
            score += float(q_val * k_val) * scale;
        }

        score *= rsqrtf(float(HEAD_DIM));
        if constexpr (CAUSAL) {
            if (kv_token > q_token) {
                score = -INFINITY;
            }
        }

        max_score = fmaxf(max_score, score);
    }

    // Phase 2: compute the softmax denominator.
    float denom = 0.0f;
    for (int kv_token = 0; kv_token < kv_len; kv_token++) {
        const int k_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * HEAD_DIM;
        const int sk_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * SCALE_DIM;

        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            const int group = d / 32;
            const int q_val = int(Q[q_offset + d]);
            const int k_val = int(K[k_offset + d]);
            const float scale = S_Q[sq_offset + group] * S_K[sk_offset + group];
            score += float(q_val * k_val) * scale;
        }

        score *= rsqrtf(float(HEAD_DIM));
        if constexpr (CAUSAL) {
            if (kv_token > q_token) {
                score = -INFINITY;
            }
        }

        denom += expf(score - max_score);
    }

    // Phase 3: use softmax probabilities to mix V into each output channel.
    for (int out_d = 0; out_d < HEAD_DIM; out_d++) {
        const int v_group = out_d / 32;
        float acc = 0.0f;

        for (int kv_token = 0; kv_token < kv_len; kv_token++) {
            const int k_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * HEAD_DIM;
            const int sk_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * SCALE_DIM;
            const int v_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * HEAD_DIM;
            const int sv_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * SCALE_DIM;

            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                const int score_group = d / 32;
                const int q_val = int(Q[q_offset + d]);
                const int k_val = int(K[k_offset + d]);
                const float scale = S_Q[sq_offset + score_group] * S_K[sk_offset + score_group];
                score += float(q_val * k_val) * scale;
            }

            score *= rsqrtf(float(HEAD_DIM));
            if constexpr (CAUSAL) {
                if (kv_token > q_token) {
                    score = -INFINITY;
                }
            }

            const float p = expf(score - max_score) / denom;
            const float v_real = float(V[v_offset + out_d]) * S_V[sv_offset + v_group];
            acc += p * v_real;
        }

        O[o_offset + out_d] = int8_attention_from_float<T>(acc);
    }
}

template <typename T, bool CAUSAL, int HEAD_DIM>
static void launch_int8_attention(
    const int8_t *Q, const int8_t *K, const int8_t *V,
    const float *S_Q, const float *S_K, const float *S_V,
    T *O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads)
{
    constexpr int INT8_HEAD_DIM = HEAD_DIM;
    constexpr int SCALE_DIM = HEAD_DIM / 32;
    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int WARP_Q = 16;
    constexpr int NUM_WARPS = BLOCK_Q / WARP_Q;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;

    const int num_blocks = bs * num_q_heads * q_len;

    constexpr int q_phase_smem = BLOCK_Q * INT8_HEAD_DIM * sizeof(int8_t) + BLOCK_Q * SCALE_DIM * sizeof(float);
    constexpr int v_phase_smem = BLOCK_KV * INT8_HEAD_DIM * sizeof(int8_t) + BLOCK_KV * SCALE_DIM * sizeof(float);
    constexpr int k_phase_smem = BLOCK_KV * INT8_HEAD_DIM * sizeof(int8_t) + BLOCK_KV * SCALE_DIM * sizeof(float);

    constexpr int smem_size = q_phase_smem + v_phase_smem;

    auto kernel = int8_attention_kernel<T, CAUSAL, BLOCK_Q, BLOCK_KV, HEAD_DIM, INT8_HEAD_DIM, SCALE_DIM, NUM_WARPS, WARP_Q>;

    kernel<<<num_blocks, 1>>>(
        Q, K, V, S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
}

template <typename T, bool CAUSAL>
static void dispatch_int8_attention(
    const int8_t *Q,
    const int8_t *K,
    const int8_t *V,
    const float *S_Q,
    const float *S_K,
    const float *S_V,
    T *O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    if (head_dim == 64)
    {
        launch_int8_attention<T, CAUSAL, 64>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    }
    else if (head_dim == 128)
    {
        launch_int8_attention<T, CAUSAL, 128>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    }
    else
    {
        fprintf(stderr, "int8_attention: unsupported head_dim=%d\n", head_dim);
    }
}

template <typename T, bool CAUSAL>
static void int8_attention_typed(
    const void *Q_raw,
    const void *K_raw,
    const void *V_raw,
    const void *S_Q_raw,
    const void *S_K_raw,
    const void *S_V_raw,
    void *O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    auto Q = reinterpret_cast<const int8_t *>(Q_raw);
    auto K = reinterpret_cast<const int8_t *>(K_raw);
    auto V = reinterpret_cast<const int8_t *>(V_raw);
    auto S_Q = reinterpret_cast<const float*>(S_Q_raw);
    auto S_K = reinterpret_cast<const float*>(S_K_raw);
    auto S_V = reinterpret_cast<const float*>(S_V_raw);
    auto O = reinterpret_cast<T*>(O_raw);

    dispatch_int8_attention<T, CAUSAL>(
        Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
        kv_capacity, num_q_heads, num_kv_heads, head_dim);
}

void int8_attention_noncausal(
    const void *Q_raw,
    const void *K_raw,
    const void *V_raw,
    const void *S_Q_raw,
    const void *S_K_raw,
    const void *S_V_raw,
    void *O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16)
{
    if (is_bf16)
    {
        int8_attention_typed<__nv_bfloat16, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
    else
    {
        int8_attention_typed<half, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}