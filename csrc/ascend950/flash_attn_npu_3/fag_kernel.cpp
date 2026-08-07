/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward device-kernel entrypoint.
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "fag_common.h"
#include "fag_epilogue_scaled_mask_softmax.hpp"
#include "fag_epilogue_sub_mul.hpp"
#include "kernel_operator.h"

static constexpr uint32_t TASK_PINGPONG = 2;
static constexpr uint8_t CROSS_CORE_SYNC_MODE = 4;
static constexpr uint16_t V0_V1_FLAG_ID_OFFSET = 16;
static constexpr uint16_t SYNC_C1_TO_V1_FLAG[TASK_PINGPONG] = {0, 1};
static constexpr uint16_t SYNC_C2_TO_V2_FLAG[TASK_PINGPONG] = {2, 3};
static constexpr uint16_t SYNC_V2_TO_C34_FLAG = 4;
static constexpr uint16_t SYNC_V1_TO_C5_FLAG = 5;
static constexpr uint16_t SYNC_C34_TO_V2_FLAG = 8;
static constexpr uint16_t SYNC_C5_TO_V1_FLAG = 9;

/*
 * FAG arch35 task pipeline and L1-buffer ownership:
 *
 *   C1: mm(Q, K^T)
 *       -- C1_TO_V1[taskId % 2] -->
 *   V1: scaledMaskSoftmax
 *       -- V1_TO_C5 --> C5: mm(P^T, dY)
 *       <-- C5_TO_V1 -- return the P L1 buffer
 *
 *   C2: mm(dY, V^T)
 *       -- C2_TO_V2[taskId % 2] -->
 *   V2: dS
 *       -- V2_TO_C34 --> C3: mm(dS, K) and C4: mm(dS^T, Q)
 *       <-- C34_TO_V2 -- return the dS L1 buffer
 *
 * In steady state, C1/C2 of task i overlap V1/V2/C5/C3/C4 of task i - 1.
 * C1/C2 completion flags use taskId % TASK_PINGPONG. P and dS L1 buffers
 * use a forward ready flag plus a reverse return flag to enforce ownership.
 */

template <
    typename DataType,
    class EpilogueScaledMaskSoftmax_,
    class EpilogueSubMul_,
    FAGTiling950::Layout INPUT_LAYOUT,
    bool IS_ATTEN_MASK,
    bool IS_DTM,
    bool IS_SOFTCAP
>
class FlashAttentionScoreGrad950 {
public:
    using EpilogueScaledMaskSoftmax =
        EpilogueScaledMaskSoftmax_;
    using EpilogueSubMul = EpilogueSubMul_;

    // Methods
    CATLASS_DEVICE
    FlashAttentionScoreGrad950() {}

    CATLASS_DEVICE
    ~FlashAttentionScoreGrad950() {}

    CATLASS_DEVICE
    void Init(FAGKernelParams const &params)
    {
        tiling_ = reinterpret_cast<const __gm__ TilingData *>(params.tiling);

        doutGm_.SetGlobalBuffer((__gm__ DataType *)params.dout);
        qGm_.SetGlobalBuffer((__gm__ DataType *)params.q);
        kGm_.SetGlobalBuffer((__gm__ DataType *)params.k);
        vGm_.SetGlobalBuffer((__gm__ DataType *)params.v);
        outGm_.SetGlobalBuffer((__gm__ DataType *)params.out);
        attenMaskGm_.SetGlobalBuffer((__gm__ uint8_t *)params.attenMask);
        softmaxLseGm_.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        cuSeqQGm_.SetGlobalBuffer((__gm__ int32_t *)params.cuSeqQlen);
        cuSeqKvGm_.SetGlobalBuffer((__gm__ int32_t *)params.cuSeqKvlen);
        dqGm_.SetGlobalBuffer((__gm__ DataType *)params.dq);
        dkGm_.SetGlobalBuffer((__gm__ DataType *)params.dk);
        dvGm_.SetGlobalBuffer((__gm__ DataType *)params.dv);

        dqWorkspaceGm_.SetGlobalBuffer(
            (__gm__ float *)(params.workspace + tiling_->dqOffset));
        dkWorkspaceGm_.SetGlobalBuffer(
            (__gm__ float *)(params.workspace + tiling_->dkOffset));
        dvWorkspaceGm_.SetGlobalBuffer(
            (__gm__ float *)(params.workspace + tiling_->dvOffset));
        deltaWorkspaceGm_.SetGlobalBuffer(
            (__gm__ float *)(params.workspace + tiling_->deltaOffset));

        batchNum_ = static_cast<uint32_t>(tiling_->batch);
        qSeqlen_ = static_cast<uint32_t>(tiling_->qSeqlen);
        kvSeqlen_ = static_cast<uint32_t>(tiling_->kvSeqlen);
        qHeadNum_ = tiling_->qHeadNum;
        kvHeadNum_ = tiling_->kvHeadNum;
        groupNum_ = static_cast<uint32_t>(tiling_->groupSize);
        qkHeadDim_ = tiling_->qkHeadDim;
        vHeadDim_ = tiling_->vHeadDim;
        qBlockSize_ = tiling_->qTile;
        kvBlockSize_ = tiling_->kvTile;
        coreNum_ = tiling_->usedCoreNum;
        continuousBlockNum_ = tiling_->continuousBlockNum;
        waveSize_ =
            static_cast<uint64_t>(coreNum_) * continuousBlockNum_;
        scaleValue_ = tiling_->scaleValue;
        softcapValue_ = tiling_->softcapValue;

        // L1 layout:
        //   [P ping][P pong][dS ping][dS pong]
        const uint32_t l1TileBytes =
            qBlockSize_ * kvBlockSize_ * sizeof(DataType);
        const uint32_t l1PBaseOffset = 0;
        const uint32_t l1dSBaseOffset =
            TASK_PINGPONG * l1TileBytes;
        for (uint32_t i = 0; i < TASK_PINGPONG; ++i) {
            l1PTensor[i] =
                resource.l1Buf.template GetBufferByByte<DataType>(
                    l1PBaseOffset + i * l1TileBytes);
            l1dSTensor[i] =
                resource.l1Buf.template GetBufferByByte<DataType>(
                    l1dSBaseOffset + i * l1TileBytes);
        }

        // Per-AIV UB layout.  Ping and pong each own one complete half of UB:
        //   ping: [mm1Res][mm2Res][attenMask][LSE][P(NZ, M+1 padding)]
        //   pong: [mm1Res][mm2Res][attenMask][LSE][P(NZ, M+1 padding)]
        // Fixpipe SPLIT_M sends half of the logical M rows to each AIV.
        const uint32_t mAligned = RoundUp(qBlockSize_, 16);
        const uint32_t rowsPerSubBlock = mAligned / 2;
        const uint32_t mmResTileBytes =
            rowsPerSubBlock * kvBlockSize_ * sizeof(float);
        const uint32_t attenMaskTileBytes =
            rowsPerSubBlock * kvBlockSize_ * sizeof(uint8_t);
        const uint32_t lseTileBytes = RoundUp(rowsPerSubBlock, 8) * sizeof(float);
        const uint32_t pTileBytes =
            (rowsPerSubBlock + 1) * kvBlockSize_ * sizeof(DataType);

        const uint32_t mm1SlotOffset = 0;
        const uint32_t mm2SlotOffset = mm1SlotOffset + mmResTileBytes;
        const uint32_t attenMaskSlotOffset = mm2SlotOffset + mmResTileBytes;
        const uint32_t lseSlotOffset = RoundUp(attenMaskSlotOffset + attenMaskTileBytes, 32);
        const uint32_t pSlotOffset = RoundUp(lseSlotOffset + lseTileBytes, 32);
        const uint32_t pingPongHalfBytes = RoundUp(pSlotOffset + pTileBytes, 32);

        for (uint32_t i = 0; i < TASK_PINGPONG; ++i) {
            const uint32_t halfBaseOffset = i * pingPongHalfBytes;
            ubMm1ResTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(
                    halfBaseOffset + mm1SlotOffset);
            ubMm2ResTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(
                    halfBaseOffset + mm2SlotOffset);
            attenMaskUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<uint8_t>(
                    halfBaseOffset + attenMaskSlotOffset);
            lseUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(
                    halfBaseOffset + lseSlotOffset);
            ubPTensor[i] =
                resource.ubBuf.template GetBufferByByte<DataType>(
                    halfBaseOffset + pSlotOffset);
        }

        if (batchNum_ != 0 && qBlockSize_ != 0 && kvBlockSize_ != 0) {
            LoadDecoderBatch();
        }
        InitEvents();
    }

    CATLASS_DEVICE
    void operator()(FAGKernelParams const &params)
    {
        Init(params);
        uint32_t coreIdx = 0;
        uint32_t subBlockIdx = 0;
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif
#ifdef __DAV_VEC__
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t vectorBlockIdx = AscendC::GetBlockIdx();
        coreIdx = vectorBlockIdx / subBlockNum;
        subBlockIdx = vectorBlockIdx % subBlockNum;
        epilogueScaledMaskSoftmax_.Init(
            resource,
            params.tiling);
        epilogueSubMul_.Init(
            resource,
            params.workspace,
            params.tiling);
        // TODO-------------------
        // AscendC::TPipe pipePre;
        // EpilogueFAGPre epilogueFagPre(xxxx);
        // epilogueFagPre();
        // pipePre.Destroy();

        // AscendC::TPipe pipeSoftmaxGrad;
        // EpilogueFAGSfmg epilogueFagSfmg(xxxx);
        // epilogueFagSfmg();
        // pipeSoftmaxGrad.Destroy();
#endif
        AscendC::SyncAll<false>();
        RunTasks(coreIdx, subBlockIdx);
        AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
        // TODO-------------------
        // AscendC::TPipe pipePost;
        // EpilogueFAGPost epilogueFagPost(xxxx);
        // epilogueFagPost();
        // pipePost.Destroy();
#endif
    }

private:
    using TilingData = FAGTiling950::FAGTilingData;

    CATLASS_DEVICE
    void InitEvents()
    {
        vWaitMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
        vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
        vWaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        vWaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
    }

    CATLASS_DEVICE
    void GetBatchShape(
        uint32_t batchIdx,
        uint64_t &qBatchStart,
        uint64_t &kvBatchStart,
        uint32_t &s1Length,
        uint32_t &s2Length)
    {
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            const uint64_t qEnd =
                static_cast<uint64_t>(cuSeqQGm_.GetValue(batchIdx));
            const uint64_t kvEnd =
                static_cast<uint64_t>(cuSeqKvGm_.GetValue(batchIdx));
            qBatchStart =
                batchIdx == 0
                    ? 0
                    : static_cast<uint64_t>(
                          cuSeqQGm_.GetValue(batchIdx - 1));
            kvBatchStart =
                batchIdx == 0
                    ? 0
                    : static_cast<uint64_t>(
                          cuSeqKvGm_.GetValue(batchIdx - 1));
            s1Length = static_cast<uint32_t>(qEnd - qBatchStart);
            s2Length = static_cast<uint32_t>(kvEnd - kvBatchStart);
        } else {
            s1Length = qSeqlen_;
            s2Length = kvSeqlen_;
            qBatchStart = static_cast<uint64_t>(batchIdx) * s1Length;
            kvBatchStart = static_cast<uint64_t>(batchIdx) * s2Length;
        }
    }

    CATLASS_DEVICE
    uint32_t FirstValidS1Block(
        uint32_t s1Length,
        uint32_t s2Length,
        uint32_t s2BlockIdx)
    {
        if constexpr (!IS_ATTEN_MASK) {
            return 0;
        }

        // floor((s2BlockIdx * kvBlockSize - (s2Length - s1Length)) /
        //       qBlockSize), clamped to [0, ceil(s1Length / qBlockSize)].
        const int64_t numerator =
            static_cast<int64_t>(s2BlockIdx) * kvBlockSize_ -
            (static_cast<int64_t>(s2Length) - s1Length);
        if (numerator <= 0) {
            return 0;
        }
        const uint32_t s1BlockNum =
            static_cast<uint32_t>(CeilDiv(s1Length, qBlockSize_));
        return Min(
            static_cast<uint32_t>(numerator / qBlockSize_), s1BlockNum);
    }

    CATLASS_DEVICE
    uint64_t CountValidS1Blocks(
        uint32_t s1Length,
        uint32_t s2Length,
        uint32_t s2BlockNum)
    {
        const uint32_t s1BlockNum =
            static_cast<uint32_t>(CeilDiv(s1Length, qBlockSize_));
        uint64_t count = 0;
        for (uint32_t s2BlockIdx = 0; s2BlockIdx < s2BlockNum;
             ++s2BlockIdx) {
            count += s1BlockNum -
                FirstValidS1Block(
                    s1Length, s2Length, s2BlockIdx);
        }
        return count;
    }

    CATLASS_DEVICE
    void LoadDecoderBatch()
    {
        GetBatchShape(
            decoderBatchIdx_,
            decoderQBatchStart_, decoderKvBatchStart_,
            decoderS1Length_, decoderS2Length_);
        decoderS1BlockNum_ = static_cast<uint32_t>(
            CeilDiv(decoderS1Length_, qBlockSize_));
        decoderS2BlockNum_ = static_cast<uint32_t>(
            CeilDiv(decoderS2Length_, kvBlockSize_));
        decoderValidS1PerN2_ = CountValidS1Blocks(
            decoderS1Length_, decoderS2Length_, decoderS2BlockNum_);
        decoderBatchBlockNum_ =
            kvHeadNum_ * groupNum_ *
            decoderValidS1PerN2_;
        decoderN2Idx_ = 0;
        decoderS2BlockIdx_ = 0;
        decoderS2BlockBegin_ = 0;
    }

    CATLASS_DEVICE
    void FillBlockInfo(
        uint64_t blockId,
        uint32_t n2Idx,
        uint32_t groupIdx,
        uint32_t s1BlockIdx,
        FAGBlockInfo &block)
    {
        const uint32_t s1Start = s1BlockIdx * qBlockSize_;
        const uint32_t s2Start = decoderS2BlockIdx_ * kvBlockSize_;
        const uint64_t totalS1Start =
            decoderQBatchStart_ + s1Start;
        const uint64_t totalS2Start =
            decoderKvBatchStart_ + s2Start;
        const uint64_t qHeadIdx =
            static_cast<uint64_t>(n2Idx) * groupNum_ + groupIdx;

        block.blockId = blockId;
        block.batchIdx = decoderBatchIdx_;
        block.n2Idx = n2Idx;
        block.groupIdx = groupIdx;
        block.s1BlockIdx = s1BlockIdx;
        block.s2BlockIdx = decoderS2BlockIdx_;
        block.s1Start = s1Start;
        block.s2Start = s2Start;
        block.curBatchS1 = decoderS1Length_;
        block.curBatchS2 = decoderS2Length_;
        block.s1Extend =
            Min(qBlockSize_, decoderS1Length_ - s1Start);
        block.s2Extend =
            Min(kvBlockSize_, decoderS2Length_ - s2Start);
        block.totalS1Start = totalS1Start;
        block.totalS2Start = totalS2Start;
        block.qOffset =
            (totalS1Start * qHeadNum_ + qHeadIdx) * qkHeadDim_;
        block.kOffset =
            (totalS2Start * kvHeadNum_ + n2Idx) * qkHeadDim_;
        block.vOffset =
            (totalS2Start * kvHeadNum_ + n2Idx) * vHeadDim_;
        block.doutOffset =
            (totalS1Start * qHeadNum_ + qHeadIdx) * vHeadDim_;
        // TODO: M方向划分和MM1输出一致
        block.firstHalfRealS1 = CeilDiv(block.s1Extend, 2U);
    }

    CATLASS_DEVICE
    bool DecodeBlock(
        uint64_t blockId,
        FAGBlockInfo &block)
    {
        // ------------bIdx------------
        while (decoderBatchIdx_ < batchNum_ &&
               blockId >= decoderBatchBlockBegin_ + decoderBatchBlockNum_) {
            decoderBatchBlockBegin_ += decoderBatchBlockNum_;
            ++decoderBatchIdx_;
            if (decoderBatchIdx_ < batchNum_) {
                LoadDecoderBatch();
            }
        }
        if (decoderBatchIdx_ >= batchNum_) {
            return false;
        }
        // ------------n2Idx------------
        const uint64_t blockNumPerN2 =
            static_cast<uint64_t>(groupNum_) * decoderValidS1PerN2_;
        const uint64_t blockInBatch =
            blockId - decoderBatchBlockBegin_;
        const uint32_t n2Idx =
            static_cast<uint32_t>(blockInBatch / blockNumPerN2);
        const uint64_t blockInN2 = blockInBatch % blockNumPerN2;
        if (n2Idx != decoderN2Idx_) {
            decoderN2Idx_ = n2Idx;
            decoderS2BlockIdx_ = 0;
            decoderS2BlockBegin_ = 0;
        }
        // ------------s2Idx------------
        uint32_t firstValidS1 = 0;
        uint32_t validS1Num = 0;
        while (decoderS2BlockIdx_ < decoderS2BlockNum_) {
            firstValidS1 = FirstValidS1Block(
                decoderS1Length_, decoderS2Length_,
                decoderS2BlockIdx_);
            validS1Num = decoderS1BlockNum_ - firstValidS1;
            const uint64_t s2BlockTasks =
                static_cast<uint64_t>(groupNum_) * validS1Num;
            if (blockInN2 <
                decoderS2BlockBegin_ + s2BlockTasks) {
                break;
            }
            decoderS2BlockBegin_ += s2BlockTasks;
            ++decoderS2BlockIdx_;
        }
        if (decoderS2BlockIdx_ >= decoderS2BlockNum_) {
            return false;
        }
        // ------------gIdx & s1Idx------------
        const uint64_t blockInS2 =
            blockInN2 - decoderS2BlockBegin_;
        const uint32_t groupIdx =
            static_cast<uint32_t>(blockInS2 / validS1Num);
        const uint32_t s1BlockIdx =
            firstValidS1 +
            static_cast<uint32_t>(blockInS2 % validS1Num);
        FillBlockInfo(
            blockId, n2Idx, groupIdx, s1BlockIdx, block);
        return true;
    }

    CATLASS_DEVICE
    void RunTasks(
        uint32_t coreIdx,
        uint32_t subBlockIdx)
    {
        if (coreIdx >= coreNum_ || coreNum_ == 0 ||
            continuousBlockNum_ == 0 ||
            qBlockSize_ == 0 || kvBlockSize_ == 0) {
            return;
        }

        uint64_t taskId = 0;
        for (uint32_t issueRound = 0;; ++issueRound) {
            const uint64_t blockBegin =
                static_cast<uint64_t>(issueRound) * waveSize_ +
                static_cast<uint64_t>(coreIdx) * continuousBlockNum_;

            for (uint32_t issueLane = 0;
                 issueLane < continuousBlockNum_; ++issueLane) {
                FAGBlockInfo block{};
                if (!DecodeBlock(blockBegin + issueLane, block)) {
                    if (taskId != 0) {
                        // Drain the last task. There is no following task, so
                        // its L1 buffers do not need to be returned.
#ifdef __DAV_CUBE__
                        ProcessC5Stage(previousBlock_, false);
                        ProcessC34Stage(previousBlock_, false);
#endif
#ifdef __DAV_VEC__
                        ProcessV1Stage(previousBlock_, subBlockIdx);
                        ProcessV2Stage(previousBlock_, subBlockIdx);
#endif
                    }
                    return;
                }
                block.taskId = taskId;
                block.issueRound = issueRound;
                block.issueLane = issueLane;

#ifdef __DAV_CUBE__
                // Front-end MM of task i overlaps the vector and back-end MM
                // stages of task i - 1.
                ProcessC1Stage(block);
                ProcessC2Stage(block);
                if (taskId != 0) {
                    ProcessC5Stage(previousBlock_, true);
                    ProcessC34Stage(previousBlock_, true);
                }
#endif
#ifdef __DAV_VEC__
                if (taskId != 0) {
                    ProcessV1Stage(previousBlock_, subBlockIdx);
                    ProcessV2Stage(previousBlock_, subBlockIdx);
                }
#endif

                previousBlock_ = block;
                ++taskId;
            }
        }
    }

#ifdef __DAV_CUBE__
    CATLASS_DEVICE
    void ProcessC1Stage(FAGBlockInfo const &block)
    {
        // C1 actual computation and its completion notification stay together.
        // TODO: Run C1 = Q * K^T for this block.
        const uint32_t flagSlot =
            static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        const uint16_t flagId = SYNC_C1_TO_V1_FLAG[flagSlot];
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
            flagId + V0_V1_FLAG_ID_OFFSET);
    }

    CATLASS_DEVICE
    void ProcessC2Stage(FAGBlockInfo const &block)
    {
        // C2 actual computation and its completion notification stay together.
        // TODO: Run C2 = dY * V^T for this block.
        const uint32_t flagSlot =
            static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        const uint16_t flagId = SYNC_C2_TO_V2_FLAG[flagSlot];
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(flagId);
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(
            flagId + V0_V1_FLAG_ID_OFFSET);
    }

    CATLASS_DEVICE
    void ProcessC34Stage(
        FAGBlockInfo const &block,
        bool returnL1)
    {
        // Wait until both vector sub-blocks have produced dS in L1.
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V2_TO_C34_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V2_TO_C34_FLAG + V0_V1_FLAG_ID_OFFSET);

        // TODO: Run C3 = dS * K for this block.
        // TODO: Run C4 = dS^T * Q for this block.

        // A following V2 stage needs an explicit ownership return before it
        // can overwrite the single dS L1 buffer. The drained last task does not.
        if (returnL1) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C34_TO_V2_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C34_TO_V2_FLAG + V0_V1_FLAG_ID_OFFSET);
        }
    }

    CATLASS_DEVICE
    void ProcessC5Stage(
        FAGBlockInfo const &block,
        bool returnL1)
    {
        // Wait until both vector sub-blocks have produced P in L1.
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V1_TO_C5_FLAG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
            SYNC_V1_TO_C5_FLAG + V0_V1_FLAG_ID_OFFSET);

        // TODO: Run C5 = P^T * dY for this block.

        // A following V1 stage needs an explicit ownership return before it
        // can overwrite the single P L1 buffer. The drained last task does not.
        if (returnL1) {
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C5_TO_V1_FLAG);
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                SYNC_C5_TO_V1_FLAG + V0_V1_FLAG_ID_OFFSET);
        }
    }
#endif

#ifdef __DAV_VEC__
    CATLASS_DEVICE
    void ProcessV1Stage(
        FAGBlockInfo const &block,
        uint32_t subBlockIdx)
    {
        const uint32_t flagSlot =
            static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(
            SYNC_C1_TO_V1_FLAG[flagSlot]);

        // Before task 1 and later overwrite P in L1, C5 must return ownership
        // of the buffer used by the preceding task.
        if (block.taskId != 0) {
            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
                SYNC_C5_TO_V1_FLAG);
        }

        auto attenMaskLayout = tla::MakeLayout<
            uint8_t, Catlass::layout::RowMajor>(256, 256);
        auto attenMaskGmTensor = tla::MakeTensor(
            attenMaskGm_, attenMaskLayout, Catlass::Arch::PositionGM{});

        uint64_t lseRows = 0;
        uint64_t lseColumns = 0;
        uint64_t lseRowOffset = 0;
        uint64_t lseColOffset = 0;
        const uint64_t n1Idx = static_cast<uint64_t>(block.n2Idx) * groupNum_ + block.groupIdx;
        const uint64_t subBlockS1Offset = subBlockIdx * block.firstHalfRealS1;
        const uint64_t subBlockRealS1 = subBlockIdx ?
            block.s1Extend - block.firstHalfRealS1 :
            block.firstHalfRealS1;
        if constexpr (INPUT_LAYOUT == FAGTiling950::Layout::TND) {
            // TND:  [N1, totalS1]
            lseRows = qHeadNum_;
            lseColumns = tiling_->totalQ;
            lseRowOffset = n1Idx;
            lseColOffset = block.totalS1Start + subBlockS1Offset;
        } else {
            // BSND: [B * N1, S1]
            lseRows = static_cast<uint64_t>(batchNum_) * qHeadNum_;
            lseColumns = qSeqlen_;
            lseRowOffset = static_cast<uint64_t>(block.batchIdx) * qHeadNum_ + n1Idx;
            lseColOffset = block.s1Start + subBlockS1Offset;
        }
        auto softmaxLseLayout = tla::MakeLayout<
            float, Catlass::layout::RowMajor>(lseRows, lseColumns);
        auto softmaxLseGmTensor = tla::MakeTensor(
            softmaxLseGm_, softmaxLseLayout,
            tla::MakeCoord(lseRowOffset, lseColOffset),
            Catlass::Arch::PositionGM{});

        event_t mte2ToVEvent = flagSlot ? vWaitMte2Pong : vWaitMte2Ping;
        event_t vToMte3Event = flagSlot ? vWaitMte3Pong : vWaitMte3Ping;

        epilogueScaledMaskSoftmax_(
            block,
            attenMaskGmTensor,
            softmaxLseGmTensor,
            attenMaskUbTensor[flagSlot],
            lseUbTensor[flagSlot],
            ubMm1ResTensor[flagSlot],
            ubPTensor[flagSlot],
            l1PTensor[flagSlot],
            mte2ToVEvent,
            vToMte3Event,
            scaleValue_,
            softcapValue_,
            subBlockRealS1,
            subBlockIdx);

        // Notify C5 only after this vector sub-block has finished writing P.
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
            SYNC_V1_TO_C5_FLAG);
    }

    CATLASS_DEVICE
    void ProcessV2Stage(
        FAGBlockInfo const &block,
        uint32_t subBlockIdx)
    {
        const uint32_t flagSlot =
            static_cast<uint32_t>(block.taskId % TASK_PINGPONG);
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(
            SYNC_C2_TO_V2_FLAG[flagSlot]);

        // Before task 1 and later overwrite dS in L1, C3/C4 must return
        // ownership of the buffer used by the preceding task.
        if (block.taskId != 0) {
            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
                SYNC_C34_TO_V2_FLAG);
        }
        const uint32_t bufferId = flagSlot;
        epilogueSubMul_(
            block,
            ubMm2ResTensor[bufferId],
            l1PTensor[bufferId],
            l1dSTensor[bufferId],
            subBlockIdx);

        // Notify C3/C4 only after this vector sub-block has finished writing dS.
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
            SYNC_V2_TO_C34_FLAG);
    }
#endif

private:
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource;
#ifdef __DAV_VEC__
    EpilogueScaledMaskSoftmax epilogueScaledMaskSoftmax_;
    EpilogueSubMul epilogueSubMul_;
#endif

    const __gm__ TilingData *tiling_ = nullptr;

    AscendC::GlobalTensor<DataType> doutGm_;
    AscendC::GlobalTensor<DataType> qGm_;
    AscendC::GlobalTensor<DataType> kGm_;
    AscendC::GlobalTensor<DataType> vGm_;
    AscendC::GlobalTensor<DataType> outGm_;
    AscendC::GlobalTensor<uint8_t> attenMaskGm_;
    AscendC::GlobalTensor<float> softmaxLseGm_;
    AscendC::GlobalTensor<int32_t> cuSeqQGm_;
    AscendC::GlobalTensor<int32_t> cuSeqKvGm_;
    AscendC::GlobalTensor<DataType> dqGm_;
    AscendC::GlobalTensor<DataType> dkGm_;
    AscendC::GlobalTensor<DataType> dvGm_;
    AscendC::GlobalTensor<float> dqWorkspaceGm_;
    AscendC::GlobalTensor<float> dkWorkspaceGm_;
    AscendC::GlobalTensor<float> dvWorkspaceGm_;
    AscendC::GlobalTensor<float> deltaWorkspaceGm_;

    AscendC::LocalTensor<DataType> l1PTensor[TASK_PINGPONG];
    AscendC::LocalTensor<DataType> l1dSTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> ubMm1ResTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> ubMm2ResTensor[TASK_PINGPONG];
    AscendC::LocalTensor<uint8_t> attenMaskUbTensor[TASK_PINGPONG];
    AscendC::LocalTensor<float> lseUbTensor[TASK_PINGPONG];
    AscendC::LocalTensor<DataType> ubPTensor[TASK_PINGPONG];

    event_t vWaitMte2Ping;
    event_t vWaitMte2Pong;
    event_t vWaitMte3Ping;
    event_t vWaitMte3Pong;

    uint32_t batchNum_ = 0;
    uint32_t qSeqlen_ = 0;
    uint32_t kvSeqlen_ = 0;
    uint64_t qHeadNum_ = 0;
    uint64_t kvHeadNum_ = 0;
    uint32_t groupNum_ = 0;
    uint64_t qkHeadDim_ = 0;
    uint64_t vHeadDim_ = 0;
    uint32_t qBlockSize_ = 0;
    uint32_t kvBlockSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t continuousBlockNum_ = 0;
    uint64_t waveSize_ = 0;
    float scaleValue_ = 1.0f;
    float softcapValue_ = 0.0f;

    uint32_t decoderBatchIdx_ = 0;
    uint64_t decoderBatchBlockBegin_ = 0;
    uint64_t decoderBatchBlockNum_ = 0;
    uint64_t decoderQBatchStart_ = 0;
    uint64_t decoderKvBatchStart_ = 0;
    uint32_t decoderS1Length_ = 0;
    uint32_t decoderS2Length_ = 0;
    uint32_t decoderS1BlockNum_ = 0;
    uint32_t decoderS2BlockNum_ = 0;
    uint64_t decoderValidS1PerN2_ = 0;

    uint32_t decoderN2Idx_ = 0;
    uint32_t decoderS2BlockIdx_ = 0;
    uint64_t decoderS2BlockBegin_ = 0;

    FAGBlockInfo previousBlock_{};
};

template <
    typename DataType,
    FAGTiling950::Layout INPUT_LAYOUT,
    bool IS_CAUSAL,
    bool IS_DETERMINISTIC,
    bool IS_SOFTCAP>
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

    using DispatchPolicyScaledMaskSoftmax =
        Catlass::Epilogue::EpilogueAscend950FAGScaledMaskSoftmax<
            INPUT_LAYOUT, IS_CAUSAL, IS_SOFTCAP>;
    using EpilogueScaledMaskSoftmax =
        Catlass::Epilogue::Block::BlockEpilogue<
            DispatchPolicyScaledMaskSoftmax,
            DataType,
            float,
            FAGTiling950::FAGTilingData>;
    using DispatchPolicySubMul =
        Catlass::Epilogue::EpilogueAscend950FAGSubMul<INPUT_LAYOUT>;
    using EpilogueSubMul =
        Catlass::Epilogue::Block::BlockEpilogue<
            DispatchPolicySubMul,
            DataType,
            DataType,
            float,
            FAGTiling950::FAGTilingData>;

    using FAGKernel950 = FlashAttentionScoreGrad950<
        DataType,
        EpilogueScaledMaskSoftmax,
        EpilogueSubMul,
        INPUT_LAYOUT,
        IS_CAUSAL,
        IS_DETERMINISTIC,
        IS_SOFTCAP>;
    FAGKernelParams params{dout, q, k, v, out, mask, softmax_lse,
        cu_seqlens_q, cu_seqlens_k, dq, dk, dv, workspace, tiling};
    FAGKernel950 fag;
    fag(params);
}
