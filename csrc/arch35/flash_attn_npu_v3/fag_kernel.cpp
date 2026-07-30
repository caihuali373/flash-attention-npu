/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward device-kernel entrypoint.
 */

#include "catlass/catlass.hpp"
#include "fag_common.h"
#include "kernel_operator.h"

template <
    FAGTiling950::Layout INPUT_LAYOUT,
    bool IS_ATTEN_MASK,
    bool IS_DTM
>
class FlashAttentionScoreGrad950 {
public:
    // Methods
    CATLASS_DEVICE
    FlashAttentionScoreGrad950() {}

    CATLASS_DEVICE
    ~FlashAttentionScoreGrad950() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(FAGKernelParams const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(FAGKernelParams const &params)
    {
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(FAGKernelParams const &params)
    {
    }

private:
};

template <
    typename DataType,
    FAGTiling950::Layout INPUT_LAYOUT,
    bool IS_CAUSAL,
    bool IS_DETERMINISTIC>
CATLASS_GLOBAL void FlashAttentionV3Bwd950(
    GM_ADDR dout,
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR out,
    GM_ADDR mask,
    GM_ADDR softmax_lse,
    GM_ADDR cu_seqlens_q,
    GM_ADDR cu_seqlens_k,
    GM_ADDR dq,
    GM_ADDR dk,
    GM_ADDR dv,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // TODO: 各个block待补充

    using FAGKernel950 = FlashAttentionScoreGrad950<
        INPUT_LAYOUT, IS_CAUSAL, IS_DETERMINISTIC>; // TODO: 待补充
    FAGKernelParams params{dout, q, k, v, out, mask, softmax_lse,
        cu_seqlens_q, cu_seqlens_k, dq, dk, dv, workspace, tiling};
    FAGKernel950 fag;
    fag(params);
}
