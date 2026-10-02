# Plan: Fix MTP and mmproj Memory Over-reservation

## Problem Statement

The `--fit` system over-reserves memory for MTP spec models and mmproj, causing unnecessary context reduction. Two key issues:

1. **MTP pre-fit chicken-and-egg**: The pre-fit uses a margin that excludes the spec reservation, resulting in a larger fitted n_ctx, which then leads to a larger spec reservation.

2. **MTP compute buffer double-counting**: The reservation adds the target model's compute buffer to the MTP's compute buffer, but they share the same buffer at runtime (`cparams.ctx_other = ctx_tgt`).

## Current Behavior

From user's logs (Qwen3.5-35B-A3B-MTP-IQ3_S):

| Metric | With mmproj | Without mmproj |
|--------|-------------|----------------|
| ROCm0 target | 4357 MiB | 3248 MiB |
| Need to reduce | 2531 MiB | 1421 MiB |
| Final n_ctx | 47,872 | 135,680 |
| Spec reservation | 4464 MiB | 4464 MiB |
| Actual spec usage | ~275 MiB | ~610 MiB |

The spec reservation is **7-16x larger** than actual usage.

## Root Causes

### Cause 1: Pre-fit margin (server-context.cpp:1385-1392)

```cpp
// Pre-fit uses params_base.fit_params_target which does NOT include spec reservation
const common_params_fit_status fit_status = common_fit_params(
    params_base.model.path.c_str(), &mparams_prefit, &cparams_prefit,
    tensor_split_prefit.data(), tensor_buft_overrides_prefit.data(),
    &moe_cache_prefit, params_base.fit_params_target.data(),  // <-- no spec yet
    params_base.fit_params_min_ctx, GGML_LOG_LEVEL_ERROR);
if (fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS) {
    cparams_dft.n_ctx = cparams_prefit.n_ctx;  // <-- too large
}
```

The margin at this point = user_target + mmproj (if any). Missing: spec reservation.

### Cause 2: Compute buffer double-counting (server-context.cpp:1408-1422)

```cpp
// For MTP: add the target model's compute buffer to the MTP measurement
const auto target = common_get_device_memory_data(...);
for (size_t i = 0; i < data.size(); i++) {
    data[i].compute += target[i].compute;  // <-- DOUBLE COUNT
}
```

At runtime (speculative.cpp:2412):
```cpp
llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
// cparams.ctx_other = ctx_tgt  (line 2385)
```

The MTP context shares the target model's compute buffer via `ctx_other`. Adding it again is wrong.

## Proposed Fixes

### Fix 1: Iterate pre-fit until margin converges

Instead of a single pre-fit, iterate:

```
1. margin = user_target + mmproj
2. Pre-fit with margin -> get n_ctx
3. Measure spec reservation at n_ctx -> get spec_bytes
4. margin = user_target + mmproj + spec_bytes
5. If |spec_bytes - prev_spec_bytes| > threshold, goto 2
6. Use converged n_ctx for final measurement
```

This breaks the chicken-and-egg by converging on the correct margin.

### Fix 2: Remove compute buffer double-counting

Simply don't add `target[i].compute` to `data[i].compute` for MTP. The MTP's own compute buffer is already measured in the first step.

### Fix 3: Use runtime n_ctx for MTP measurement (alternative)

Instead of iterating, use the actual n_ctx that will be used at runtime:
- At runtime, `cparams.n_ctx = llama_n_ctx(ctx_tgt)` (speculative.cpp:2378)
- The target's n_ctx is already fitted by the main fit
- So MTP should use the same n_ctx

This is simpler but requires passing the target's fitted n_ctx to the measurement.

## Files to Modify

1. `tools/server/server-context.cpp` - Fix pre-fit iteration and compute buffer
2. `common/fit.cpp` - May need to expose n_ctx from main fit

## Testing

1. Run with Qwen3.5-35B-A3B-MTP-IQ3_S (no mmproj) - should get ~135k context (no change expected)
2. Run with Qwen3.5-35B-A3B-MTP-IQ3_S + mmproj - should get significantly more context than 47k
3. Verify spec reservation is closer to actual usage (~275-610 MiB instead of 4464 MiB)

## Risk Assessment

- **Low risk**: Fix 2 is a simple removal of incorrect double-counting
- **Medium risk**: Fix 1 adds iteration, but the logic is straightforward
- **Alternative**: Fix 3 is simplest but may not handle all edge cases

Recommend starting with Fix 2 (simplest, most impactful), then Fix 1 or Fix 3.
