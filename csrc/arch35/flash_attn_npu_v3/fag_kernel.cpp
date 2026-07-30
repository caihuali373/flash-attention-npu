/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward device-kernel entrypoint.
 */

#include "catlass/catlass.hpp"
#include "fag_tilingdata.h"
#include "kernel_operator.h"

CATLASS_GLOBAL void FlashAttentionV3Bwd950(
    GM_ADDR dout,
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR out,
    GM_ADDR softmax_lse,
    GM_ADDR cu_seqlens_q,
    GM_ADDR cu_seqlens_k,
    GM_ADDR dq,
    GM_ADDR dk,
    GM_ADDR dv,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    auto *fag_tiling =
        reinterpret_cast<__gm__ FAGTiling950::FAGTilingData *>(tiling);
    if (AscendC::GetBlockIdx() == 0) {
        AscendC::printf(
            "[A5 FAG V3] entered FlashAttentionV3Bwd950, "
            "block_num=%u layout=%d\n",
            AscendC::GetBlockNum(), fag_tiling->layout);
        AscendC::printf("totalQ=%d totalKv=%d N1=%d N2=%d G=%d D1=%d D2=%d\n",
            fag_tiling->totalQ, fag_tiling->totalKv,
            fag_tiling->qHeadNum, fag_tiling->kvHeadNum,
            fag_tiling->groupSize, fag_tiling->qkHeadDim, fag_tiling->vHeadDim);
    }

    (void)workspace;

    // TODO(arch35-bwd): implement the Ascend950 backward kernel.
}
