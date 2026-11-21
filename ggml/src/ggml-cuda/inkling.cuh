#pragma once

#include "common.cuh"

// Fused kernels for the Inkling 2 graph. Each replaces a chain of
// small elementwise ggml nodes that otherwise cost one kernel launch apiece.

// dst = y * clamp(fill / (smul*||y|| + sbias), lo, hi) per row, with y = x (* scale)
void ggml_cuda_ink_l2_clip(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * scale,
        float smul, float sbias, float fill, float lo, float hi, ggml_tensor * dst);

// dst = go*tanh(gi*g)*sigmoid(g) * uo*tanh(ui*u)
void ggml_cuda_ink_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
        float gi, float go, float ui, float uo, ggml_tensor * dst);

// dst = cur - tanh(f)*cur + tanh(g)*msg
void ggml_cuda_ink_ctrl(ggml_backend_cuda_context & ctx, const ggml_tensor * f, const ggml_tensor * g,
        const ggml_tensor * cur, const ggml_tensor * msg, ggml_tensor * dst);

// depth attention over the rows of a (and b, when the states are concat(a, b, 1)):
// S = rms_norm(states [E,K,T]), p = softmax_k(scale * S^T q), dst[E,NQ,T] = S p
bool ggml_cuda_ink_mix_supported(int K, int NQ);
void ggml_cuda_ink_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * query,
        float eps, float kq_scale, ggml_tensor * dst);

// dst = scale*(cur + ssm_conv(sx, w))
void ggml_cuda_ink_conv_res(ggml_backend_cuda_context & ctx, const ggml_tensor * sx, const ggml_tensor * w, const ggml_tensor * cur,
        float scale, ggml_tensor * dst);

// GGML_OP_INK_REL_BIAS: the per-head attention mask from the relative-position bias table
void ggml_cuda_op_ink_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dst = scale*(cur + ssm_conv(sx, w) + input-dependent taps per head)
void ggml_cuda_ink_dconv_res(ggml_backend_cuda_context & ctx, const ggml_tensor * sx, const ggml_tensor * w, const ggml_tensor * taps,
        const ggml_tensor * cur, float scale, ggml_tensor * dst);

// out = concat(past, xt, 0) and the last past->ne[0] columns of out into state
void ggml_cuda_ink_cat_store(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * xt, ggml_tensor * out,
        ggml_tensor * state);

// dst = concat(a, rms_norm(b), 0)
void ggml_cuda_ink_cat_rms(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b, float eps, ggml_tensor * dst);
