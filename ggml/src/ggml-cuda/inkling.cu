#include "inkling.cuh"

// ---------------------------------------------------------------- l2 clip
// one warp per row
static __global__ void k_ink_l2_clip(const float * __restrict__ x, const float * __restrict__ scale, float * dst,
        const int ne0, const int ne1, const int sne1, const int nrows,
        const float smul, const float sbias, const float fill, const float lo, const float hi) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const float * xr = x + (size_t) row*ne0;
    const float * sr = scale ? scale + (size_t) ((row % ne1) % sne1)*ne0 : nullptr;
    float       * dr = dst + (size_t) row*ne0;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < ne0; i += WARP_SIZE) {
        float v = xr[i];
        if (sr) {
            v *= sr[i];
        }
        ss += v*v;
    }
    ss = warp_reduce_sum(ss);

    const float norm = smul*sqrtf(ss) + sbias;
    const float f    = fminf(fmaxf(fill/norm, lo), hi);

    for (int i = threadIdx.x; i < ne0; i += WARP_SIZE) {
        float v = xr[i];
        if (sr) {
            v *= sr[i];
        }
        dr[i] = v*f;
    }
}

void ggml_cuda_ink_l2_clip(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * scale,
        float smul, float sbias, float fill, float lo, float hi, ggml_tensor * dst) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
    const int ne0   = x->ne[0];
    const int ne1   = x->ne[1];
    const int nrows = ggml_nrows(x);
    int sne1 = 1;
    if (scale) {
        GGML_ASSERT(scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) && scale->ne[0] == ne0);
        sne1 = scale->ne[1];
    }
    const dim3 block(WARP_SIZE, 4, 1);
    const dim3 grid((nrows + 3)/4, 1, 1);
    k_ink_l2_clip<<<grid, block, 0, ctx.stream()>>>((const float *) x->data, scale ? (const float *) scale->data : nullptr,
            (float *) dst->data, ne0, ne1, sne1, nrows, smul, sbias, fill, lo, hi);
}

// ---------------------------------------------------------------- situ
static __global__ void k_ink_situ(const char * __restrict__ gate, const char * __restrict__ up, float * dst,
        const int ne0, const int ne1, const int ne2,
        const size_t gnb1, const size_t gnb2, const size_t unb1, const size_t unb2,
        const float gi, const float go, const float ui, const float uo) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t n = (int64_t) ne0*ne1*ne2;
    if (i >= n) {
        return;
    }
    const int i0 = i % ne0;
    const int i1 = (i / ne0) % ne1;
    const int i2 = i / ((int64_t) ne0*ne1);
    const float g = ((const float *) (gate + i1*gnb1 + i2*gnb2))[i0];
    const float u = ((const float *) (up   + i1*unb1 + i2*unb2))[i0];
    const float act = (go*tanhf(gi*g)) * (1.0f/(1.0f + expf(-g)));
    dst[i] = act * (uo*tanhf(ui*u));
}

void ggml_cuda_ink_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
        float gi, float go, float ui, float uo, ggml_tensor * dst) {
    GGML_ASSERT(gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(gate->nb[0] == sizeof(float) && up->nb[0] == sizeof(float) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(gate, up) && ggml_are_same_shape(gate, dst) && gate->ne[3] == 1);
    const int64_t n = ggml_nelements(dst);
    const int bs = 256;
    k_ink_situ<<<(n + bs - 1)/bs, bs, 0, ctx.stream()>>>((const char *) gate->data, (const char *) up->data, (float *) dst->data,
            gate->ne[0], gate->ne[1], gate->ne[2], gate->nb[1], gate->nb[2], up->nb[1], up->nb[2], gi, go, ui, uo);
}

// ---------------------------------------------------------------- loop controller tail
static __global__ void k_ink_ctrl(const float * __restrict__ f, const float * __restrict__ g,
        const float * cur, const float * __restrict__ msg, float * dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float c = cur[i];
    dst[i] = (c - tanhf(f[i])*c) + tanhf(g[i])*msg[i];
}

void ggml_cuda_ink_ctrl(ggml_backend_cuda_context & ctx, const ggml_tensor * f, const ggml_tensor * g,
        const ggml_tensor * cur, const ggml_tensor * msg, ggml_tensor * dst) {
    for (const ggml_tensor * t : { f, g, cur, msg, (const ggml_tensor *) dst }) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_are_same_shape(t, dst));
    }
    const int64_t n = ggml_nelements(dst);
    const int bs = 256;
    k_ink_ctrl<<<(n + bs - 1)/bs, bs, 0, ctx.stream()>>>((const float *) f->data, (const float *) g->data,
            (const float *) cur->data, (const float *) msg->data, (float *) dst->data, n);
}

// ---------------------------------------------------------------- depth attention (mix)
#define INK_MIX_MAX_K  32
#define INK_MIX_MAX_NQ 4
#define INK_MIX_NT     1024

// one block per token; NQ is a template parameter so every per-query accumulator stays in registers.
// Pass 1 splits each row over several warps so many loads are in flight at once (latency, not bandwidth, bound).
template <int NQ>
static __global__ void __launch_bounds__(INK_MIX_NT) k_ink_mix(const float * __restrict__ A, const float * __restrict__ B,
        const float * __restrict__ query, float * dst, const int E, const int Ka, const int K, const float eps, const float kq_scale) {
    const int t    = blockIdx.x;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    constexpr int nwarps = INK_MIX_NT / WARP_SIZE;

    const int Kb = K - Ka;
    // row k of token t
    auto row = [&](int k) -> const float * {
        return k < Ka ? A + ((size_t) t*Ka + k)*E : B + ((size_t) t*Kb + (k - Ka))*E;
    };
    float * D = dst + (size_t) t*E*NQ;

    __shared__ float red[INK_MIX_MAX_K][NQ + 1];
    __shared__ float sc[NQ][INK_MIX_MAX_K];

    for (int i = threadIdx.x; i < INK_MIX_MAX_K*(NQ + 1); i += INK_MIX_NT) {
        (&red[0][0])[i] = 0.0f;
    }
    __syncthreads();

    // pass 1: warps are split into (row, segment) pairs
    const int nseg = nwarps / K > 0 ? nwarps / K : 1;
    for (int w = warp; w < K*nseg; w += nwarps) {
        const int k   = w / nseg;
        const int seg = w % nseg;
        const int len = (E + nseg - 1) / nseg;
        const int e0  = seg*len;
        const int e1  = min(E, e0 + len);
        const float * s = row(k);
        float ss = 0.0f;
        float d[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            d[q] = 0.0f;
        }
#pragma unroll 4
        for (int e = e0 + lane; e < e1; e += WARP_SIZE) {
            const float v = s[e];
            ss += v*v;
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                d[q] += v*query[(size_t) q*E + e];
            }
        }
        ss = warp_reduce_sum(ss);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            d[q] = warp_reduce_sum(d[q]);
        }
        if (lane == 0) {
            atomicAdd(&red[k][0], ss);
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                atomicAdd(&red[k][q + 1], d[q]);
            }
        }
    }
    __syncthreads();

    // softmax over k, one thread per query; the row normalization folds into the weights
    if (threadIdx.x < NQ) {
        const int q = threadIdx.x;
        float * p = sc[q];
        float mx = -INFINITY;
        for (int k = 0; k < K; ++k) {
            p[k] = red[k][q + 1]*rsqrtf(red[k][0]/E + eps)*kq_scale;
            mx = fmaxf(mx, p[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float e = expf(p[k] - mx);
            p[k] = e;
            sum += e;
        }
        const float is = 1.0f/sum;
        for (int k = 0; k < K; ++k) {
            p[k] *= is*rsqrtf(red[k][0]/E + eps);
        }
    }
    __syncthreads();

    // pass 2: weighted sum of the rows
    for (int e = threadIdx.x; e < E; e += INK_MIX_NT) {
        float acc[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            acc[q] = 0.0f;
        }
#pragma unroll 4
        for (int k = 0; k < K; ++k) {
            const float v = row(k)[e];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                acc[q] += v*sc[q][k];
            }
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            D[(size_t) q*E + e] = acc[q];
        }
    }
}

// float4 variant: E/4 is a multiple of the warp size, so each warp stays inside one row in pass 1
template <int NQ>
static __global__ void __launch_bounds__(INK_MIX_NT) k_ink_mix_v4(const float4 * __restrict__ A, const float4 * __restrict__ B,
        const float4 * __restrict__ query, float4 * dst, const int E4, const int Ka, const int K, const float eps, const float kq_scale) {
    const int t    = blockIdx.x;
    const int lane = threadIdx.x % WARP_SIZE;
    const int Kb   = K - Ka;
    auto row = [&](int k) -> const float4 * {
        return k < Ka ? A + ((size_t) t*Ka + k)*E4 : B + ((size_t) t*Kb + (k - Ka))*E4;
    };
    float4 * D = dst + (size_t) t*E4*NQ;

    __shared__ float red[INK_MIX_MAX_K][NQ + 1];
    __shared__ float sc[NQ][INK_MIX_MAX_K];
    for (int i = threadIdx.x; i < INK_MIX_MAX_K*(NQ + 1); i += INK_MIX_NT) {
        (&red[0][0])[i] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < K*E4; idx += INK_MIX_NT) {
        const int k  = idx / E4;
        const int e4 = idx % E4;
        const float4 v = row(k)[e4];
        float ss = v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        float d[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const float4 w = query[(size_t) q*E4 + e4];
            d[q] = v.x*w.x + v.y*w.y + v.z*w.z + v.w*w.w;
        }
        ss = warp_reduce_sum(ss);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            d[q] = warp_reduce_sum(d[q]);
        }
        if (lane == 0) {
            atomicAdd(&red[k][0], ss);
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                atomicAdd(&red[k][q + 1], d[q]);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < NQ) {
        const int q = threadIdx.x;
        const float E = 4.0f*E4;
        float * p = sc[q];
        float mx = -INFINITY;
        for (int k = 0; k < K; ++k) {
            p[k] = red[k][q + 1]*rsqrtf(red[k][0]/E + eps)*kq_scale;
            mx = fmaxf(mx, p[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float e = expf(p[k] - mx);
            p[k] = e;
            sum += e;
        }
        const float is = 1.0f/sum;
        for (int k = 0; k < K; ++k) {
            p[k] *= is*rsqrtf(red[k][0]/E + eps);
        }
    }
    __syncthreads();

    for (int e4 = threadIdx.x; e4 < E4; e4 += INK_MIX_NT) {
        float4 acc[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            acc[q] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll 4
        for (int k = 0; k < K; ++k) {
            const float4 v = row(k)[e4];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float p = sc[q][k];
                acc[q].x += v.x*p; acc[q].y += v.y*p; acc[q].z += v.z*p; acc[q].w += v.w*p;
            }
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            D[(size_t) q*E4 + e4] = acc[q];
        }
    }
}

static bool ink_aligned16(const void * p) {
    return ((uintptr_t) p & 15) == 0;
}

bool ggml_cuda_ink_mix_supported(int K, int NQ) {
    return K <= INK_MIX_MAX_K && NQ >= 1 && NQ <= INK_MIX_MAX_NQ;
}

void ggml_cuda_ink_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * query,
        float eps, float kq_scale, ggml_tensor * dst) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 && query->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(query) && ggml_is_contiguous(dst));
    const int E  = a->ne[0];
    const int Ka = a->ne[1];
    const int T  = a->ne[2]*a->ne[3];
    int K = Ka;
    if (b) {
        GGML_ASSERT(b->type == GGML_TYPE_F32 && ggml_is_contiguous(b) && b->ne[0] == E && b->ne[2]*b->ne[3] == T);
        K += b->ne[1];
    }
    const int NQ = query->ne[1];
    GGML_ASSERT(query->ne[0] == E && dst->ne[0] == E && dst->ne[1] == NQ && dst->ne[2]*dst->ne[3] == T);
    GGML_ASSERT(ggml_cuda_ink_mix_supported(K, NQ));
    const float * pa = (const float *) a->data;
    const float * pb = b ? (const float *) b->data : nullptr;
    const float * q  = (const float *) query->data;
    float       * d  = (float *) dst->data;
    cudaStream_t st = ctx.stream();
    if (E % (4*WARP_SIZE) == 0 && ink_aligned16(pa) && (!pb || ink_aligned16(pb)) && ink_aligned16(q) && ink_aligned16(d)) {
        const float4 * a4 = (const float4 *) pa;
        const float4 * b4 = (const float4 *) pb;
        const float4 * q4 = (const float4 *) q;
        float4       * d4 = (float4 *) d;
        switch (NQ) {
            case 1: k_ink_mix_v4<1><<<T, INK_MIX_NT, 0, st>>>(a4, b4, q4, d4, E/4, Ka, K, eps, kq_scale); return;
            case 2: k_ink_mix_v4<2><<<T, INK_MIX_NT, 0, st>>>(a4, b4, q4, d4, E/4, Ka, K, eps, kq_scale); return;
            case 3: k_ink_mix_v4<3><<<T, INK_MIX_NT, 0, st>>>(a4, b4, q4, d4, E/4, Ka, K, eps, kq_scale); return;
            case 4: k_ink_mix_v4<4><<<T, INK_MIX_NT, 0, st>>>(a4, b4, q4, d4, E/4, Ka, K, eps, kq_scale); return;
            default: GGML_ABORT("unsupported NQ");
        }
    }
    switch (NQ) {
        case 1: k_ink_mix<1><<<T, INK_MIX_NT, 0, st>>>(pa, pb, q, d, E, Ka, K, eps, kq_scale); break;
        case 2: k_ink_mix<2><<<T, INK_MIX_NT, 0, st>>>(pa, pb, q, d, E, Ka, K, eps, kq_scale); break;
        case 3: k_ink_mix<3><<<T, INK_MIX_NT, 0, st>>>(pa, pb, q, d, E, Ka, K, eps, kq_scale); break;
        case 4: k_ink_mix<4><<<T, INK_MIX_NT, 0, st>>>(pa, pb, q, d, E, Ka, K, eps, kq_scale); break;
        default: GGML_ABORT("unsupported NQ");
    }
}

// ---------------------------------------------------------------- static short convolution + residual + scale
// sx [n_conv - 1 + n_t, C, n_s] (time innermost), w [n_conv, C], cur/dst [C, n_t*n_s]
static __global__ void k_ink_conv_res(const char * __restrict__ sx, const float * __restrict__ w, const float * __restrict__ cur, float * dst,
        const int n_conv, const int C, const int n_t, const int n_s, const size_t sx_nb1, const size_t sx_nb2,
        const int64_t cur_nb1, const float scale) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) C*n_t*n_s) {
        return;
    }
    const int c  = i % C;
    const int ts = i / C;
    const int t  = ts % n_t;
    const int s  = ts / n_t;
    const float * x  = (const float *) (sx + c*sx_nb1 + s*sx_nb2) + t;
    const float * wc = w + (size_t) c*n_conv;
    float acc = 0.0f;
    for (int k = 0; k < n_conv; ++k) {
        acc += x[k]*wc[k];
    }
    dst[i] = scale*(cur[(size_t) ts*cur_nb1 + c] + acc);
}

void ggml_cuda_ink_conv_res(ggml_backend_cuda_context & ctx, const ggml_tensor * sx, const ggml_tensor * w, const ggml_tensor * cur,
        float scale, ggml_tensor * dst) {
    GGML_ASSERT(sx->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && cur->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(sx->nb[0] == sizeof(float) && ggml_is_contiguous(w) && ggml_is_contiguous(dst) && cur->nb[0] == sizeof(float));
    const int n_conv = w->ne[0];
    const int C      = w->ne[1];
    const int n_t    = sx->ne[0] - n_conv + 1;
    const int n_s    = sx->ne[2];
    GGML_ASSERT(sx->ne[1] == C && ggml_nelements(dst) == (int64_t) C*n_t*n_s && cur->ne[0] == C);
    const int64_t n = ggml_nelements(dst);
    const int bs = 256;
    k_ink_conv_res<<<(n + bs - 1)/bs, bs, 0, ctx.stream()>>>((const char *) sx->data, (const float *) w->data, (const float *) cur->data,
            (float *) dst->data, n_conv, C, n_t, n_s, sx->nb[1], sx->nb[2], cur->nb[1]/sizeof(float), scale);
}

// ---------------------------------------------------------------- GGML_OP_INK_REL_BIAS
// dst[j, t', h, s] = table[idx[j, t], h, t] - sink[h, t] + mask[j, t', 0, s], t = s*n_tps + t'
template <typename TM, typename TD>
static __global__ void k_ink_rel_bias(const char * __restrict__ table, const float * __restrict__ sink, const char * __restrict__ idx,
        const char * __restrict__ mask, char * dst, const int n_kv, const int n_tps, const int n_head, const int n_str,
        const size_t t_nb0, const size_t t_nb1, const size_t t_nb2, const size_t i_nb1,
        const size_t m_nb1, const size_t m_nb3, const size_t d_nb1, const size_t d_nb2, const size_t d_nb3) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) n_kv*n_tps*n_head*n_str) {
        return;
    }
    const int j  = i % n_kv;
    const int tp = (i / n_kv) % n_tps;
    const int h  = (i / ((int64_t) n_kv*n_tps)) % n_head;
    const int st = i / ((int64_t) n_kv*n_tps*n_head);
    const int t  = st*n_tps + tp;

    const int r = ((const int32_t *) (idx + t*i_nb1))[j];
    const TM  mv = ((const TM *) (mask + tp*m_nb1 + st*m_nb3))[j];
    float m;
    if constexpr (std::is_same_v<TM, half>) {
        m = __half2float(mv);
    } else {
        m = mv;
    }
    const float v = *(const float *) (table + r*t_nb0 + h*t_nb1 + t*t_nb2) - sink[h + (size_t) n_head*t] + m;
    TD * d = (TD *) (dst + tp*d_nb1 + h*d_nb2 + st*d_nb3) + j;
    if constexpr (std::is_same_v<TD, half>) {
        *d = __float2half(v);
    } else {
        *d = v;
    }
}

template <typename TM, typename TD>
static void ink_rel_bias_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table = dst->src[0];
    const ggml_tensor * sinks = dst->src[1];
    const ggml_tensor * idx   = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const int64_t n = ggml_nelements(dst);
    const int bs = 256;
    k_ink_rel_bias<TM, TD><<<(n + bs - 1)/bs, bs, 0, ctx.stream()>>>((const char *) table->data, (const float *) sinks->data,
            (const char *) idx->data, (const char *) mask->data, (char *) dst->data, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            table->nb[0], table->nb[1], table->nb[2], idx->nb[1], mask->nb[1], mask->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]);
}

void ggml_cuda_op_ink_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mask = dst->src[3];
    GGML_ASSERT(dst->src[2]->nb[0] == sizeof(int32_t) && mask->nb[0] == ggml_type_size(mask->type) && dst->nb[0] == ggml_type_size(dst->type));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    if (mask->type == GGML_TYPE_F16) {
        dst->type == GGML_TYPE_F16 ? ink_rel_bias_launch<half, half>(ctx, dst) : ink_rel_bias_launch<half, float>(ctx, dst);
    } else {
        dst->type == GGML_TYPE_F16 ? ink_rel_bias_launch<float, half>(ctx, dst) : ink_rel_bias_launch<float, float>(ctx, dst);
    }
}

// ---------------------------------------------------------------- dynamic short convolution + residual + scale
// out[c, t] = scale*(cur[c, t] + sum_k w[c, k]*x[t + k, c] + sum_l taps[l + n_conv*h(c), t]*x[t + n_conv - 1 - l, c])
static __global__ void k_ink_dconv_res(const char * __restrict__ sx, const float * __restrict__ w, const float * __restrict__ taps,
        const float * __restrict__ cur, float * dst, const int n_conv, const int C, const int Dh, const int n_t,
        const size_t sx_nb1, const int64_t taps_nb1, const int64_t cur_nb1, const float scale) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) C*n_t) {
        return;
    }
    const int c = i % C;
    const int t = i / C;
    const int h = c / Dh;
    const float * x  = (const float *) (sx + c*sx_nb1) + t;
    const float * wc = w + (size_t) c*n_conv;
    const float * tp = taps + (size_t) t*taps_nb1 + (size_t) n_conv*h;
    float acc = 0.0f;
    for (int k = 0; k < n_conv; ++k) {
        acc += x[k]*wc[k];
    }
    for (int l = 0; l < n_conv; ++l) {
        acc += tp[l]*x[n_conv - 1 - l];
    }
    dst[i] = scale*(cur[(size_t) t*cur_nb1 + c] + acc);
}

void ggml_cuda_ink_dconv_res(ggml_backend_cuda_context & ctx, const ggml_tensor * sx, const ggml_tensor * w, const ggml_tensor * taps,
        const ggml_tensor * cur, float scale, ggml_tensor * dst) {
    GGML_ASSERT(sx->type == GGML_TYPE_F32 && sx->nb[0] == sizeof(float) && sx->ne[2] == 1);
    GGML_ASSERT(ggml_is_contiguous(w) && taps->type == GGML_TYPE_F32 && taps->nb[0] == sizeof(float));
    GGML_ASSERT(cur->type == GGML_TYPE_F32 && cur->nb[0] == sizeof(float) && ggml_is_contiguous(dst));
    const int n_conv = w->ne[0];
    const int C      = w->ne[1];
    const int n_t    = sx->ne[0] - n_conv + 1;
    const int H      = taps->ne[0] / n_conv;
    GGML_ASSERT(sx->ne[1] == C && H > 0 && C % H == 0 && ggml_nelements(dst) == (int64_t) C*n_t);
    const int64_t n = (int64_t) C*n_t;
    const int bs = 256;
    k_ink_dconv_res<<<(n + bs - 1)/bs, bs, 0, ctx.stream()>>>((const char *) sx->data, (const float *) w->data, (const float *) taps->data,
            (const float *) cur->data, (float *) dst->data, n_conv, C, C/H, n_t, sx->nb[1], taps->nb[1]/sizeof(float),
            cur->nb[1]/sizeof(float), scale);
}

// ---------------------------------------------------------------- conv input concat + recurrent state store
// out[tau, c] = tau < P ? past[tau, c] : xt[tau - P, c]; state[k + P*c] = out[T + k, c] for k < P
static __global__ void k_ink_cat_store(const char * __restrict__ past, const char * __restrict__ xt, float * out, float * state,
        const int P, const int T, const int C, const size_t p_nb0, const size_t p_nb1, const size_t x_nb0, const size_t x_nb1) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int L = P + T;
    if (i >= (int64_t) L*C) {
        return;
    }
    const int tau = i % L;
    const int c   = i / L;
    const float v = tau < P ? *(const float *) (past + tau*p_nb0 + c*p_nb1) : *(const float *) (xt + (tau - P)*x_nb0 + c*x_nb1);
    out[i] = v;
    if (tau >= T) {
        state[(tau - T) + (size_t) P*c] = v;
    }
}

void ggml_cuda_ink_cat_store(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * xt, ggml_tensor * out,
        ggml_tensor * state) {
    GGML_ASSERT(past->type == GGML_TYPE_F32 && xt->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F32 && state->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(out) && ggml_is_contiguous(state));
    const int P = past->ne[0], T = xt->ne[0], C = past->ne[1];
    GGML_ASSERT(xt->ne[1] == C && out->ne[0] == P + T && out->ne[1] == C && ggml_nelements(state) == (int64_t) P*C);
    const int64_t n = (int64_t) (P + T)*C;
    const int bs = 256;
    k_ink_cat_store<<<(n + bs - 1)/bs, bs, 0, ctx.stream()>>>((const char *) past->data, (const char *) xt->data, (float *) out->data,
            (float *) state->data, P, T, C, past->nb[0], past->nb[1], xt->nb[0], xt->nb[1]);
}

// ---------------------------------------------------------------- concat(a, rms_norm(b)) along dim 0, one block per row
template <typename V>
static __global__ void __launch_bounds__(1024) k_ink_cat_rms(const V * __restrict__ a, const V * __restrict__ b, V * dst,
        const int Ea, const int Eb, const int n_per, const float eps) {
    // Ea, Eb in units of V; n_per = floats per V
    const int t = blockIdx.x;
    const V * ar = a + (size_t) t*Ea;
    const V * br = b + (size_t) t*Eb;
    V       * dr = dst + (size_t) t*(Ea + Eb);
    float ss = 0.0f;
    for (int e = threadIdx.x; e < Eb; e += blockDim.x) {
        const float * v = (const float *) &br[e];
        for (int i = 0; i < n_per; ++i) {
            ss += v[i]*v[i];
        }
    }
    __shared__ float s_red[32];
    ss = warp_reduce_sum(ss);
    if (threadIdx.x % WARP_SIZE == 0) {
        s_red[threadIdx.x / WARP_SIZE] = ss;
    }
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < (int) (blockDim.x / WARP_SIZE); ++w) {
        tot += s_red[w];
    }
    const float r = rsqrtf(tot/(Eb*n_per) + eps);
    for (int e = threadIdx.x; e < Ea; e += blockDim.x) {
        dr[e] = ar[e];
    }
    for (int e = threadIdx.x; e < Eb; e += blockDim.x) {
        V v = br[e];
        float * f = (float *) &v;
        for (int i = 0; i < n_per; ++i) {
            f[i] *= r;
        }
        dr[Ea + e] = v;
    }
}

void ggml_cuda_ink_cat_rms(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b, float eps, ggml_tensor * dst) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst));
    const int T = ggml_nrows(a);
    GGML_ASSERT(ggml_nrows(b) == T && ggml_nrows(dst) == T && dst->ne[0] == a->ne[0] + b->ne[0]);
    const int Ea = a->ne[0], Eb = b->ne[0];
    if (Ea % 4 == 0 && Eb % 4 == 0 && ink_aligned16(a->data) && ink_aligned16(b->data) && ink_aligned16(dst->data)) {
        k_ink_cat_rms<float4><<<T, 1024, 0, ctx.stream()>>>((const float4 *) a->data, (const float4 *) b->data, (float4 *) dst->data,
                Ea/4, Eb/4, 4, eps);
    } else {
        k_ink_cat_rms<float><<<T, 1024, 0, ctx.stream()>>>((const float *) a->data, (const float *) b->data, (float *) dst->data,
                Ea, Eb, 1, eps);
    }
}
