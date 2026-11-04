/*!
 * L1-custom dq/dk for D=128 (Aligned128).
 * FAG mul/drop are cast half-ND (vec), not float-NZ Cube1:
 *   dq: RowMajor Nd2Nz → zN → NoTranspose LoadData → CFG_ROW_MAJOR Fixpipe
 *   dk: ColumnMajor Nd2Nz → nZ → Transpose LoadData (match Catlass LayoutA3);
 *       sab bulk-Transpose on reused zN assumes float-NZ Cube1 and fails ~40%
 *   dv: Catlass BlockMmad3
 *
 * Partial-tile: L1 slots are MMAD_BASE_SIZE×MMAD_BASE_SIZE; Nd2Nz dstNzC0Stride
 * and LoadData srcStride use the full slot (Catlass layoutDst), Mmad uses mRound
 * (subMSizeActAlign) with m==1→2 quirk; Fixpipe mSize stays actual.
 *
 * Event partition: L1-custom M_MTE1 {4..7}; Catlass dv uses eventIdStart=0.
 * Included inside FlashAttentionScoreGrad; all L1/L0 state is local.
 */
#ifndef FAG_COMMON_L1_CUSTOM_DQKV_HPP
#define FAG_COMMON_L1_CUSTOM_DQKV_HPP

// RowMajor ND → L1 zN. l1RowsAlign = full L1 slot rows (MMAD_BASE_SIZE).
__aicore__ inline void CopyGmToL1Nd(const AscendC::LocalTensor<ElementA>& l1Tensor,
                                    const AscendC::GlobalTensor<ElementA>& gmSrcTensor, uint32_t srcN, uint32_t srcD,
                                    uint32_t srcDstride, uint32_t l1RowsAlign)
{
    AscendC::Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN;
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = l1RowsAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

// ColumnMajor ND (M×K, col stride) → L1 nZ. Matches Catlass CopyGmToL1<ColumnMajor>:
// dValue=M, nValue=K, srcDValue=col stride; dstNzC0Stride = K-align of L1 slot.
__aicore__ inline void CopyGmToL1NdCol(const AscendC::LocalTensor<ElementA>& l1Tensor,
                                       const AscendC::GlobalTensor<ElementA>& gmSrcTensor, uint32_t srcM, uint32_t srcK,
                                       uint32_t srcStride, uint32_t l1ColsAlign)
{
    AscendC::Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.dValue = srcM;
    nd2nzPara.nValue = srcK;
    nd2nzPara.srcDValue = srcStride;
    nd2nzPara.dstNzC0Stride = l1ColsAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

__aicore__ inline void ComputeL1CustomMMDqkv(DBParams& dbParam, int64_t nextBlockId)
{
    constexpr uint32_t SIZE_256 = 256;
    constexpr uint32_t SIZE_32 = 32;
    constexpr uint32_t SIZE_64 = 64;
    constexpr uint32_t SIZE_ONE_K = 1024;
    constexpr uint32_t L1_DS_BYTES = 512 * 256 * 2;
    constexpr uint32_t L1_KV_BYTES = 512 * 128 * 2;
    constexpr uint32_t L1_SLOT = MMAD_BASE_SIZE;
    constexpr uint32_t L1_SLOT_FR = L1_SLOT / C0_SIZE;

    AscendC::LocalTensor<ElementA> dsL1Tensor = resource.l1Buf.template GetBufferByByte<ElementA>(0);
    AscendC::LocalTensor<ElementA> kL1Tensor = resource.l1Buf.template GetBufferByByte<ElementA>(L1_DS_BYTES);
    AscendC::LocalTensor<ElementA> qL1Tensor =
        resource.l1Buf.template GetBufferByByte<ElementA>(L1_DS_BYTES + L1_KV_BYTES);

    AscendC::LocalTensor<ElementA> l0_a_ping_tensor = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
    AscendC::LocalTensor<ElementA> l0_a_pong_tensor =
        resource.l0ABuf.template GetBufferByByte<ElementA>(SIZE_32 * SIZE_ONE_K);
    AscendC::LocalTensor<ElementA> l0_b_ping_tensor = resource.l0BBuf.template GetBufferByByte<ElementA>(0);
    AscendC::LocalTensor<ElementA> l0_b_pong_tensor =
        resource.l0BBuf.template GetBufferByByte<ElementA>(SIZE_32 * SIZE_ONE_K);
    AscendC::LocalTensor<float> l0_c_ping_tensor = resource.l0CBuf.template GetBufferByByte<float>(0);
    AscendC::LocalTensor<float> l0_c_pong_tensor =
        resource.l0CBuf.template GetBufferByByte<float>(SIZE_64 * SIZE_ONE_K);

    AscendC::LoadData2dParams commonLoadData2dParamsTranspose{0, MMAD_BASE_SIZE, MMAD_BASE_SIZE, 0, 0, true, 0};
    AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose{0, MMAD_BASE_SIZE, MMAD_BASE_SIZE, 0, 0, false, 0};
    AscendC::MmadParams commonMadParams{MMAD_BASE_SIZE, MMAD_BASE_SIZE, MMAD_BASE_SIZE, 3, false, true};
    AscendC::FixpipeParamsV220 commonFixpipeParamsV220{MMAD_BASE_SIZE, MMAD_BASE_SIZE, MMAD_BASE_SIZE, MMAD_BASE_SIZE,
                                                       false};

    uint32_t ping_pong_flag_l1_a_ = 0;
    uint32_t ping_pong_flag_l0_a_ = 0;
    uint32_t ping_pong_flag_l0_b_ = 0;
    uint32_t ping_pong_flag_l0_c_ = 0;

    pingpongIdx = dbParam.taskId % 2;

    const int64_t dqOffset = dbParam.aTensorOffsetCv;
    const int64_t dkvOffset = dbParam.bTensorOffsetCv;
    const int64_t dvOffset = dbParam.bTensorOffsetCv / d * value_d;
    const int64_t dqKc = dbParam.s1Stride;
    const int64_t dkvKc = dbParam.s2Stride;
    const int64_t dvKc = dbParam.s2Stride / d * value_d;
    const uint64_t s2_stride_ws = static_cast<uint64_t>(dbParam.s2CvExtendAlign) * DTYPE_FACTOR;

    constexpr int32_t s2SliceSize = S_SPLITT_SIZE;
    int32_t s2SliceNum = (dbParam.s2CvExtend + s2SliceSize - 1) / s2SliceSize;
    int32_t s2SliceTail = dbParam.s2CvExtend - (s2SliceNum - 1) * s2SliceSize;

    constexpr event_t EVENT_MTE1_MTE2_DRAIN = static_cast<event_t>(2);
    constexpr event_t EVENT_MTE1_A_TO_M = static_cast<event_t>(0);
    constexpr event_t EVENT_MTE1_B_TO_M = static_cast<event_t>(1);
    constexpr event_t EVENT_M_MTE1_DRAIN = static_cast<event_t>(0);

    uint32_t mSplitSize = MMAD_BASE_SIZE;
    uint32_t mLoops = (dbParam.s1CvExtend + mSplitSize - 1) / mSplitSize;
    uint32_t mTail = dbParam.s1CvExtend - (mLoops - 1) * mSplitSize;
    uint32_t subMSizeAct = mSplitSize;
    uint32_t subMSizeActAlign = CeilDiv(subMSizeAct, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;

    uint32_t nSplitSize = MMAD_BASE_SIZE;
    uint32_t subNSizeAct = nSplitSize;
    uint32_t subNSizeActAlign = CeilDiv(subNSizeAct, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;

    constexpr uint32_t L1C_M_MTE1_BASE = 4;
    AscendC::SetLoadDataPaddingValue(static_cast<ElementA>(0));
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L1C_M_MTE1_BASE + 0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L1C_M_MTE1_BASE + 1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L1C_M_MTE1_BASE + 2);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L1C_M_MTE1_BASE + 3);

    for (int32_t sliceIdx = 0; sliceIdx < s2SliceNum; ++sliceIdx) {
        int32_t s2SliceExtend = sliceIdx == (s2SliceNum - 1) ? s2SliceTail : s2SliceSize;
        uint32_t nInnerLoops = (s2SliceExtend + nSplitSize - 1) / nSplitSize;
        uint32_t nInnerTail = s2SliceExtend - (nInnerLoops - 1) * nSplitSize;
        int64_t bTensorOffsetBias = static_cast<int64_t>(sliceIdx) * s2SliceSize * dbParam.s2Stride;
        int64_t dkOffsetBias = bTensorOffsetBias;

        ///////////////////////////////////////////////////////////////
        // Matmul4 dq: dS * K
        ///////////////////////////////////////////////////////////////
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ping_pong_flag_l1_a_);
        for (uint32_t m = 0; m < mLoops; m++) {
            subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
            subMSizeActAlign = CeilDiv(subMSizeAct, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;
            AscendC::LocalTensor<float>* l0_c_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;
            for (uint32_t n = 0; n < nInnerLoops; n++) {
                subNSizeAct = n == (nInnerLoops - 1) ? nInnerTail : nSplitSize;
                subNSizeActAlign = CeilDiv(subNSizeAct, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;
                // K reloaded every slice (m==0); offsets stay inside 128KB kL1.
                uint64_t bL1Offset = static_cast<uint64_t>(n) * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                uint64_t aL1Offset = static_cast<uint64_t>(m) * MMAD_BASE_SIZE * S_SPLITT_SIZE +
                                     static_cast<uint64_t>(MMAD_BASE_SIZE) * n * MMAD_BASE_SIZE;
                AscendC::LocalTensor<ElementA>* l0_a_tensor =
                    ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                AscendC::LocalTensor<ElementA>* l0_b_tensor =
                    ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;

                int64_t dsGmOffset = static_cast<int64_t>(pingpongIdx) * cubeBaseMN * DTYPE_FACTOR +
                                     static_cast<int64_t>(m) * mSplitSize * static_cast<int64_t>(s2_stride_ws) +
                                     static_cast<int64_t>(sliceIdx) * s2SliceSize +
                                     static_cast<int64_t>(n) * nSplitSize;
                CopyGmToL1Nd(dsL1Tensor[aL1Offset], mulWorkSpaceGm[dsGmOffset], subMSizeAct, subNSizeAct,
                             static_cast<uint32_t>(s2_stride_ws), L1_SLOT);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + L1C_M_MTE1_BASE);

                commonLoadData2dParamsNoTranspose.repeatTimes = subNSizeActAlign / C0_SIZE;
                commonLoadData2dParamsNoTranspose.srcStride = L1_SLOT_FR;
                for (int32_t i = 0; i < static_cast<int32_t>(subMSizeActAlign / C0_SIZE); i++) {
                    AscendC::LoadData((*l0_a_tensor)[i * subNSizeActAlign * C0_SIZE],
                                      dsL1Tensor[aL1Offset + i * SIZE_256], commonLoadData2dParamsNoTranspose);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_MTE1_A_TO_M);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_MTE1_A_TO_M);

                if (m == 0) {
                    CopyGmToL1Nd(kL1Tensor[bL1Offset],
                                 keyGm[dbParam.bTensorOffsetCv + bTensorOffsetBias +
                                       static_cast<int64_t>(n) * nSplitSize * dbParam.s2Stride],
                                 subNSizeAct, static_cast<uint32_t>(d), static_cast<uint32_t>(dbParam.s2Stride),
                                 L1_SLOT);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + L1C_M_MTE1_BASE + 2);
                commonLoadData2dParamsTranspose.repeatTimes = static_cast<uint16_t>(d / C0_SIZE);
                commonLoadData2dParamsTranspose.srcStride = L1_SLOT_FR;
                for (int32_t i = 0; i < static_cast<int32_t>(subNSizeActAlign / C0_SIZE); i++) {
                    AscendC::LoadData((*l0_b_tensor)[i * d * C0_SIZE], kL1Tensor[bL1Offset + i * SIZE_256],
                                      commonLoadData2dParamsTranspose);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_MTE1_B_TO_M);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_MTE1_B_TO_M);

                uint16_t m_mad = (subMSizeAct == 1) ? 2 : static_cast<uint16_t>(subMSizeActAlign);
                commonMadParams.m = m_mad;
                commonMadParams.n = static_cast<uint16_t>(d);
                commonMadParams.k = static_cast<uint16_t>(subNSizeAct);
                commonMadParams.unitFlag = n == nInnerLoops - 1 ? 3 : 2;
                commonMadParams.cmatrixInitVal = n == 0;

                if (n == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ping_pong_flag_l0_c_);
                }
                if (n == nInnerLoops - 1) {
                    AscendC::PipeBarrier<PIPE_M>();
                }
                AscendC::Mmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, commonMadParams);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + L1C_M_MTE1_BASE + 2);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + L1C_M_MTE1_BASE);
                ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
                ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
            }

            uint32_t gmBaseBlockOffset = m * mSplitSize * static_cast<uint32_t>(dqKc);
            commonFixpipeParamsV220.mSize = subMSizeAct;
            commonFixpipeParamsV220.nSize = static_cast<uint32_t>(d);
            commonFixpipeParamsV220.srcStride = subMSizeActAlign;
            commonFixpipeParamsV220.dstStride = static_cast<uint32_t>(dqKc);
            commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
            commonFixpipeParamsV220.unitFlag = 3;
            AscendC::SetAtomicType<float>();
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(dqWorkSpaceGm[dqOffset + gmBaseBlockOffset],
                                                                   *l0_c_tensor, commonFixpipeParamsV220);
            AscendC::SetAtomicNone();
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ping_pong_flag_l0_c_);
            ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ping_pong_flag_l1_a_);
        ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_MTE1_MTE2_DRAIN);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_MTE1_MTE2_DRAIN);

        ///////////////////////////////////////////////////////////////
        // Matmul4 dk: dS^T * Q — ColumnMajor Nd2Nz + nZ→L0A (Catlass)
        ///////////////////////////////////////////////////////////////
        for (uint32_t n_loop = 0; n_loop < nInnerLoops; n_loop++) {
            subNSizeAct = n_loop == (nInnerLoops - 1) ? nInnerTail : nSplitSize;
            subNSizeActAlign = CeilDiv(subNSizeAct, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;
            AscendC::LocalTensor<float>* l0_c_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;
            for (uint32_t m_loop = 0; m_loop < mLoops; m_loop++) {
                subMSizeAct = m_loop == (mLoops - 1) ? mTail : mSplitSize;
                subMSizeActAlign = CeilDiv(subMSizeAct, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;
                uint64_t bL1Offset = static_cast<uint64_t>(m_loop) * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                uint64_t aL1Offset = static_cast<uint64_t>(m_loop) * MMAD_BASE_SIZE * S_SPLITT_SIZE +
                                     static_cast<uint64_t>(n_loop) * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                AscendC::LocalTensor<ElementA>* l0_a_tensor =
                    ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                AscendC::LocalTensor<ElementA>* l0_b_tensor =
                    ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;

                int64_t dsGmOffset = static_cast<int64_t>(pingpongIdx) * cubeBaseMN * DTYPE_FACTOR +
                                     static_cast<int64_t>(m_loop) * mSplitSize * static_cast<int64_t>(s2_stride_ws) +
                                     static_cast<int64_t>(sliceIdx) * s2SliceSize +
                                     static_cast<int64_t>(n_loop) * nSplitSize;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + L1C_M_MTE1_BASE);
                // ColumnMajor view (M=s2, K=s1) → nZ; l1ColsAlign = K slot = 128
                CopyGmToL1NdCol(dsL1Tensor[aL1Offset], mulWorkSpaceGm[dsGmOffset], subNSizeAct, subMSizeAct,
                                static_cast<uint32_t>(s2_stride_ws), L1_SLOT);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);

                // nZ → L0A Transpose (Catlass CopyL1ToL0A<nZ>)
                commonLoadData2dParamsTranspose.repeatTimes = static_cast<uint16_t>(subMSizeActAlign / C0_SIZE);
                commonLoadData2dParamsTranspose.srcStride = 1;
                for (int32_t i = 0; i < static_cast<int32_t>(subNSizeActAlign / C0_SIZE); i++) {
                    AscendC::LoadData((*l0_a_tensor)[i * subMSizeActAlign * C0_SIZE],
                                      dsL1Tensor[aL1Offset + static_cast<uint64_t>(i) * L1_SLOT * C0_SIZE],
                                      commonLoadData2dParamsTranspose);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_MTE1_A_TO_M);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_MTE1_A_TO_M);

                if (n_loop == 0 && sliceIdx == 0) {
                    CopyGmToL1Nd(
                        qL1Tensor[bL1Offset],
                        queryGm[dbParam.aTensorOffsetCv + static_cast<int64_t>(m_loop) * mSplitSize * dbParam.s1Stride],
                        subMSizeAct, static_cast<uint32_t>(d), static_cast<uint32_t>(dbParam.s1Stride), L1_SLOT);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + L1C_M_MTE1_BASE + 2);
                commonLoadData2dParamsTranspose.repeatTimes = static_cast<uint16_t>(d / C0_SIZE);
                commonLoadData2dParamsTranspose.srcStride = L1_SLOT_FR;
                for (int32_t i = 0; i < static_cast<int32_t>(subMSizeActAlign / C0_SIZE); i++) {
                    AscendC::LoadData((*l0_b_tensor)[i * d * C0_SIZE], qL1Tensor[bL1Offset + i * SIZE_256],
                                      commonLoadData2dParamsTranspose);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_MTE1_B_TO_M);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_MTE1_B_TO_M);

                uint16_t s2_mad = (subNSizeAct == 1) ? 2 : static_cast<uint16_t>(subNSizeActAlign);
                commonMadParams.m = s2_mad;
                commonMadParams.n = static_cast<uint16_t>(d);
                commonMadParams.k = static_cast<uint16_t>(subMSizeAct);
                commonMadParams.unitFlag = m_loop == mLoops - 1 ? 3 : 2;
                commonMadParams.cmatrixInitVal = m_loop == 0;

                if (m_loop == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ping_pong_flag_l0_c_);
                }
                if (m_loop == mLoops - 1) {
                    AscendC::PipeBarrier<PIPE_M>();
                }
                AscendC::Mmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, commonMadParams);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + L1C_M_MTE1_BASE + 2);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + L1C_M_MTE1_BASE);
                ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
                ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
            }

            uint32_t gmBaseBlockOffset = n_loop * nSplitSize * static_cast<uint32_t>(dkvKc);
            commonFixpipeParamsV220.mSize = subNSizeAct;
            commonFixpipeParamsV220.nSize = static_cast<uint32_t>(d);
            commonFixpipeParamsV220.srcStride = subNSizeActAlign;
            commonFixpipeParamsV220.dstStride = static_cast<uint32_t>(dkvKc);
            commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
            commonFixpipeParamsV220.unitFlag = 3;
            AscendC::SetAtomicType<float>();
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                dkWorkSpaceGm[dkvOffset + dkOffsetBias + gmBaseBlockOffset], *l0_c_tensor, commonFixpipeParamsV220);
            AscendC::SetAtomicNone();
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ping_pong_flag_l0_c_);
            ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_MTE1_MTE2_DRAIN);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_MTE1_MTE2_DRAIN);
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L1C_M_MTE1_BASE + 0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L1C_M_MTE1_BASE + 1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L1C_M_MTE1_BASE + 2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L1C_M_MTE1_BASE + 3);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_M_MTE1_DRAIN);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_M_MTE1_DRAIN);

    ///////////////////////////////////////////////////////////////
    // dv via Catlass (eventIdStart=0).
    ///////////////////////////////////////////////////////////////
    int64_t s2_size = dbParam.s2CvExtendAlign * DTYPE_FACTOR;
    GemmCoord actualBlockShape3(dbParam.s2CvExtend, value_d, dbParam.s1CvExtend);
    LayoutA3 layoutA3{actualBlockShape3.m(), actualBlockShape3.k(), s2_size};
    LayoutB3 layoutB3{actualBlockShape3.k(), actualBlockShape3.n(), dbParam.s1Stride / d * value_d};
    LayoutC3 layoutC3{actualBlockShape3.m(), actualBlockShape3.n(), dvKc};
    BlockMmad3 blockMmad3Dv(resource, 0, 0);
    blockMmad3Dv(dropWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], dxGm[dbParam.aTensorOffsetCv / d * value_d],
                 dvWorkSpaceGm[dvOffset], layoutA3, layoutB3, layoutC3, actualBlockShape3, true, true);
    (void)nextBlockId;
}

#endif // FAG_COMMON_L1_CUSTOM_DQKV_HPP
