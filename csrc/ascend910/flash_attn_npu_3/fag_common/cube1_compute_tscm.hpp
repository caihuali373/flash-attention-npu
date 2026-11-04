/*!
 * sameAB Cube1Compute (TSCM) for D=64/128/256 MM1 (dyv + qk).
 *
 * Port of ops-transformer sab.h Cube1Compute (Aligned128 path), adapted for FAGGeneral:
 *   - L1 A/B reuse (B full-load across m), L0A/B/C ping-pong
 *   - CFG_ROW_MAJOR Fixpipe (FAG mm1/mm2 workspace is float ND RowMajor,
 *     stride=s2CvExtendAlign; sameAB often uses CFG_NZ)
 *   - All L1/L0/event state is local to ComputeMM1 — SyncAll-safe
 *   - Does NOT persist BlockMmad across SyncAll
 *   - Aligned64 uses 512*64*2 L1 A/B (sameAB Aligned64 uses Matmul API + asymmetric
 *     L1; we keep low-level Cube1 with equal A/B slots sized for headDim)
 *   - Aligned256: L0A ping is 32KB → cannot hold 128×256; K-split (dSplit=128)
 *     with L0C accumulate across dLoops (unitFlag 2→3). sameAB D=256 uses
 *     MatmulImpl GM; we stay on DataCopy/Mmad/Fixpipe.
 *   - P5/P6 MM1→dq K-share rolled back: (P5) next MM1 overwrites L1 before
 *     prev dq; (P6) dual L1B still FAIL 020 dq — Cube1 NZ≠Catlass zN read.
 *
 * Included inside FlashAttentionScoreGrad.
 */
#ifndef FAG_COMMON_CUBE1_COMPUTE_TSCM_HPP
#define FAG_COMMON_CUBE1_COMPUTE_TSCM_HPP

    __aicore__ inline void Cube1CopyGmToL1(const AscendC::LocalTensor<ElementA> &l1Tensor,
                                           const AscendC::GlobalTensor<ElementA> &gmSrcTensor, uint32_t srcN,
                                           uint32_t srcD, uint32_t srcDstride)
    {
        AscendC::Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = srcN;
        nd2nzPara.dValue = srcD;
        nd2nzPara.srcDValue = srcDstride;
        nd2nzPara.dstNzC0Stride = CeilDiv(srcN, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
    }

    // Scoped Cube1Compute: left[M,K] * right[N,K]^T → out[M,N] float ND.
    // ping-pong flags are shared across qk then dyv within one ComputeMM1.
    __aicore__ inline void Cube1ComputeTscm(
        const AscendC::GlobalTensor<ElementA> &left, const AscendC::GlobalTensor<ElementA> &right,
        const AscendC::GlobalTensor<float> &out, const int32_t headDim, DBParams &dbParam,
        AscendC::LocalTensor<ElementA> &aL1Tensor, AscendC::LocalTensor<ElementA> &bL1Tensor,
        AscendC::LocalTensor<ElementA> &l0_a_ping_tensor, AscendC::LocalTensor<ElementA> &l0_a_pong_tensor,
        AscendC::LocalTensor<ElementA> &l0_b_ping_tensor, AscendC::LocalTensor<ElementA> &l0_b_pong_tensor,
        AscendC::LocalTensor<float> &l0_c_ping_tensor, AscendC::LocalTensor<float> &l0_c_pong_tensor,
        AscendC::LoadData2dParams &commonLoadData2dParamsNoTranspose, AscendC::MmadParams &commonMadParams,
        AscendC::FixpipeParamsV220 &commonFixpipeParamsV220, uint32_t &ping_pong_flag_l0_a_,
        uint32_t &ping_pong_flag_l0_b_, uint32_t &ping_pong_flag_l0_c_,
        const event_t eventIdMte1AToM, const event_t eventIdMte1BToM)
    {
        constexpr uint32_t SIZE_256 = 256;
        // value_d may differ from d; scale offsets from d-based dbParam fields.
        const int64_t aTensorOffset =
            (headDim == static_cast<int32_t>(d)) ? dbParam.aTensorOffsetCv
                                                 : (dbParam.aTensorOffsetCv / d * headDim);
        const int64_t bTensorOffset =
            (headDim == static_cast<int32_t>(d)) ? dbParam.bTensorOffsetCv
                                                 : (dbParam.bTensorOffsetCv / d * headDim);
        const int64_t s1StrideHd = (headDim == static_cast<int32_t>(d)) ? dbParam.s1Stride
                                                                       : (dbParam.s1Stride / d * headDim);
        const int64_t s2StrideHd = (headDim == static_cast<int32_t>(d)) ? dbParam.s2Stride
                                                                       : (dbParam.s2Stride / d * headDim);

        uint32_t mSplitSize = MMAD_BASE_SIZE;
        uint32_t mLoops = (dbParam.s1CvExtend + mSplitSize - 1) / mSplitSize;
        uint32_t mTail = dbParam.s1CvExtend - (mLoops - 1) * mSplitSize;
        uint32_t subMSizeAct = mSplitSize;

        uint32_t nSplitSize = MMAD_BASE_SIZE;
        uint32_t nLoops = (dbParam.s2CvExtend + nSplitSize - 1) / nSplitSize;
        uint32_t nTail = dbParam.s2CvExtend - (nLoops - 1) * nSplitSize;
        uint32_t subNSizeAct = nSplitSize;

        const uint32_t headDimAlign =
            (static_cast<uint32_t>(headDim) + static_cast<uint32_t>(C0_SIZE) - 1) /
            static_cast<uint32_t>(C0_SIZE) * static_cast<uint32_t>(C0_SIZE);
        const uint32_t outRowStride = static_cast<uint32_t>(dbParam.s2CvExtendAlign);

        // L0A ping half = 32KB → max K for 128-row tile is 128. D>128 → K-split.
        const uint32_t dSplitSize =
            (headDimAlign > MMAD_BASE_SIZE) ? static_cast<uint32_t>(MMAD_BASE_SIZE) : headDimAlign;
        const uint32_t dLoops = (headDimAlign + dSplitSize - 1) / dSplitSize;
        const uint32_t dTail = headDimAlign - (dLoops - 1) * dSplitSize;

        for (uint32_t m = 0; m < mLoops; m++) {
            subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
            const uint32_t subMSizeActAlign =
                CeilDiv(subMSizeAct, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;
            const uint64_t aL1Offset = static_cast<uint64_t>(m) * mSplitSize * headDimAlign;

            Cube1CopyGmToL1(aL1Tensor[aL1Offset],
                            left[aTensorOffset + static_cast<int64_t>(m) * mSplitSize * s1StrideHd],
                            subMSizeAct, static_cast<uint32_t>(headDim),
                            static_cast<uint32_t>(s1StrideHd));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);

            if (dLoops == 1) {
                // D<=128: load full-K A once, reuse across n (existing TSCM path).
                AscendC::LocalTensor<ElementA> *l0_a_tensor =
                    ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + ping_pong_flag_l0_a_);
                commonLoadData2dParamsNoTranspose.repeatTimes =
                    static_cast<uint16_t>(headDim / static_cast<int32_t>(C0_SIZE));
                commonLoadData2dParamsNoTranspose.srcStride =
                    static_cast<uint16_t>(subMSizeActAlign / C0_SIZE);
                for (int32_t i = 0; i < static_cast<int32_t>(subMSizeActAlign / C0_SIZE); i++) {
                    AscendC::LoadData((*l0_a_tensor)[i * headDim * static_cast<int32_t>(C0_SIZE)],
                                      aL1Tensor[aL1Offset + static_cast<uint64_t>(i) * SIZE_256],
                                      commonLoadData2dParamsNoTranspose);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventIdMte1AToM);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventIdMte1AToM);

                for (uint32_t n = 0; n < nLoops; n++) {
                    subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
                    const uint32_t subNSizeActAlign =
                        CeilDiv(subNSizeAct, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;
                    const uint64_t bL1Offset = static_cast<uint64_t>(n) * nSplitSize * headDimAlign;
                    AscendC::LocalTensor<ElementA> *l0_b_tensor =
                        ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;
                    AscendC::LocalTensor<float> *l0_c_tensor =
                        ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;

                    if (m == 0) {
                        Cube1CopyGmToL1(
                            bL1Tensor[bL1Offset],
                            right[bTensorOffset + static_cast<int64_t>(n) * nSplitSize * s2StrideHd],
                            subNSizeAct, static_cast<uint32_t>(headDim),
                            static_cast<uint32_t>(s2StrideHd));
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + ping_pong_flag_l0_b_ + 2);
                    commonLoadData2dParamsNoTranspose.repeatTimes =
                        static_cast<uint16_t>(subNSizeActAlign / C0_SIZE);
                    commonLoadData2dParamsNoTranspose.srcStride = 1;
                    for (int32_t i = 0; i < static_cast<int32_t>(headDim / static_cast<int32_t>(C0_SIZE));
                         i++) {
                        AscendC::LoadData(
                            (*l0_b_tensor)[i * static_cast<int32_t>(subNSizeActAlign) *
                                           static_cast<int32_t>(C0_SIZE)],
                            bL1Tensor[bL1Offset + static_cast<uint64_t>(i) * subNSizeActAlign * C0_SIZE],
                            commonLoadData2dParamsNoTranspose);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventIdMte1BToM);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventIdMte1BToM);

                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ping_pong_flag_l0_c_);
                    {
                        const uint16_t m_modify =
                            (subMSizeAct == 1) ? 2 : static_cast<uint16_t>(subMSizeAct);
                        commonMadParams.m = m_modify;
                        commonMadParams.n = static_cast<uint16_t>(subNSizeAct);
                        commonMadParams.k = static_cast<uint16_t>(headDim);
                        commonMadParams.unitFlag = 3;
                        commonMadParams.cmatrixInitVal = true;
                        AscendC::Mmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, commonMadParams);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + 3 + 2);

                    {
                        const int64_t gmOffset =
                            static_cast<int64_t>(pingpongIdx) * cubeBaseMN +
                            static_cast<int64_t>(m) * mSplitSize * static_cast<int64_t>(outRowStride) +
                            static_cast<int64_t>(n) * nSplitSize;
                        commonFixpipeParamsV220.mSize = subMSizeAct;
                        commonFixpipeParamsV220.nSize = subNSizeAct;
                        commonFixpipeParamsV220.srcStride = subMSizeActAlign;
                        commonFixpipeParamsV220.dstStride = outRowStride;
                        commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
                        commonFixpipeParamsV220.unitFlag = 3;
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(out[gmOffset], *l0_c_tensor,
                                                                              commonFixpipeParamsV220);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ping_pong_flag_l0_c_);
                    ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
                    ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + 3);
                ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
            } else {
                // D=256: K-split into dSplitSize=128; L0C accumulates; Fixpipe once per (m,n).
                for (uint32_t n = 0; n < nLoops; n++) {
                    subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
                    const uint32_t subNSizeActAlign =
                        CeilDiv(subNSizeAct, static_cast<uint32_t>(C0_SIZE)) * C0_SIZE;
                    const uint64_t bL1Offset = static_cast<uint64_t>(n) * nSplitSize * headDimAlign;
                    AscendC::LocalTensor<float> *l0_c_tensor =
                        ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;

                    if (m == 0) {
                        Cube1CopyGmToL1(
                            bL1Tensor[bL1Offset],
                            right[bTensorOffset + static_cast<int64_t>(n) * nSplitSize * s2StrideHd],
                            subNSizeAct, static_cast<uint32_t>(headDim),
                            static_cast<uint32_t>(s2StrideHd));
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ping_pong_flag_l0_c_);
                    for (uint32_t dLoopIdx = 0; dLoopIdx < dLoops; dLoopIdx++) {
                        const uint32_t subDSizeAct = (dLoopIdx == dLoops - 1) ? dTail : dSplitSize;
                        AscendC::LocalTensor<ElementA> *l0_a_tensor =
                            ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                        AscendC::LocalTensor<ElementA> *l0_b_tensor =
                            ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + ping_pong_flag_l0_a_);
                        const uint64_t aKOffset =
                            static_cast<uint64_t>(dLoopIdx) * dSplitSize * subMSizeActAlign;
                        commonLoadData2dParamsNoTranspose.repeatTimes =
                            static_cast<uint16_t>(subDSizeAct / C0_SIZE);
                        commonLoadData2dParamsNoTranspose.srcStride =
                            static_cast<uint16_t>(subMSizeActAlign / C0_SIZE);
                        for (int32_t i = 0; i < static_cast<int32_t>(subMSizeActAlign / C0_SIZE); i++) {
                            AscendC::LoadData(
                                (*l0_a_tensor)[i * static_cast<int32_t>(subDSizeAct) *
                                               static_cast<int32_t>(C0_SIZE)],
                                aL1Tensor[aL1Offset + aKOffset + static_cast<uint64_t>(i) * SIZE_256],
                                commonLoadData2dParamsNoTranspose);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventIdMte1AToM);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventIdMte1AToM);

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + ping_pong_flag_l0_b_ + 2);
                        commonLoadData2dParamsNoTranspose.repeatTimes =
                            static_cast<uint16_t>(subNSizeActAlign / C0_SIZE);
                        commonLoadData2dParamsNoTranspose.srcStride = 1;
                        const uint64_t bKOffset =
                            static_cast<uint64_t>(dLoopIdx) * dSplitSize * subNSizeActAlign;
                        for (int32_t i = 0; i < static_cast<int32_t>(subDSizeAct / C0_SIZE); i++) {
                            AscendC::LoadData(
                                (*l0_b_tensor)[i * static_cast<int32_t>(subNSizeActAlign) *
                                               static_cast<int32_t>(C0_SIZE)],
                                bL1Tensor[bL1Offset + bKOffset +
                                          static_cast<uint64_t>(i) * subNSizeActAlign * C0_SIZE],
                                commonLoadData2dParamsNoTranspose);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventIdMte1BToM);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventIdMte1BToM);

                        {
                            const uint16_t m_modify =
                                (subMSizeAct == 1) ? 2 : static_cast<uint16_t>(subMSizeAct);
                            commonMadParams.m = m_modify;
                            commonMadParams.n = static_cast<uint16_t>(subNSizeAct);
                            commonMadParams.k = static_cast<uint16_t>(subDSizeAct);
                            commonMadParams.unitFlag = (dLoopIdx == dLoops - 1) ? 3 : 2;
                            commonMadParams.cmatrixInitVal = (dLoopIdx == 0);
                            AscendC::Mmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, commonMadParams);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + 3);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + 3 + 2);
                        ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
                        ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
                    }

                    {
                        const int64_t gmOffset =
                            static_cast<int64_t>(pingpongIdx) * cubeBaseMN +
                            static_cast<int64_t>(m) * mSplitSize * static_cast<int64_t>(outRowStride) +
                            static_cast<int64_t>(n) * nSplitSize;
                        commonFixpipeParamsV220.mSize = subMSizeAct;
                        commonFixpipeParamsV220.nSize = subNSizeAct;
                        commonFixpipeParamsV220.srcStride = subMSizeActAlign;
                        commonFixpipeParamsV220.dstStride = outRowStride;
                        commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
                        commonFixpipeParamsV220.unitFlag = 3;
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(out[gmOffset], *l0_c_tensor,
                                                                              commonFixpipeParamsV220);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ping_pong_flag_l0_c_);
                    ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
                }
            }
        }
    }

    // D=64/128/256 MM1 via Cube1Compute (TSCM). Caller only enters for those D;
    // Aligned192 keeps Catlass stack BlockMmad.
    __aicore__ inline void ComputeMM1Cube1Tscm(DBParams &dbParam)
    {
        constexpr uint32_t SIZE_32 = 32;
        constexpr uint32_t SIZE_64 = 64;
        constexpr uint32_t SIZE_ONE_K = 1024;
        // L1 A/B: 512 * headDimAlign * sizeof(ElementA); fits s1/s2CvExtend <= 512.
        // Aligned256: 512*256*2*2 = 512KB fills L1 (MM1-scoped, then drained).
        constexpr uint32_t HEAD_DIM_ALIGN =
            (DTEMPLATETYPE == DTemplateType::Aligned64)    ? 64u :
            (DTEMPLATETYPE == DTemplateType::Aligned256)   ? 256u : 128u;
        constexpr uint32_t L1_A_BYTES = 512 * HEAD_DIM_ALIGN * 2;
        constexpr uint32_t L1_B_BYTES = 512 * HEAD_DIM_ALIGN * 2;
        static_assert(L1_A_BYTES + L1_B_BYTES <= 512 * 1024,
                      "Cube1 TSCM L1 A+B must fit 512KB L1");

        AscendC::LocalTensor<ElementA> aL1Tensor =
            resource.l1Buf.template GetBufferByByte<ElementA>(0);
        AscendC::LocalTensor<ElementA> bL1Tensor =
            resource.l1Buf.template GetBufferByByte<ElementA>(L1_A_BYTES);

        AscendC::LocalTensor<ElementA> l0_a_ping_tensor =
            resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        AscendC::LocalTensor<ElementA> l0_a_pong_tensor =
            resource.l0ABuf.template GetBufferByByte<ElementA>(SIZE_32 * SIZE_ONE_K);
        AscendC::LocalTensor<ElementA> l0_b_ping_tensor =
            resource.l0BBuf.template GetBufferByByte<ElementA>(0);
        AscendC::LocalTensor<ElementA> l0_b_pong_tensor =
            resource.l0BBuf.template GetBufferByByte<ElementA>(SIZE_32 * SIZE_ONE_K);
        AscendC::LocalTensor<float> l0_c_ping_tensor =
            resource.l0CBuf.template GetBufferByByte<float>(0);
        AscendC::LocalTensor<float> l0_c_pong_tensor =
            resource.l0CBuf.template GetBufferByByte<float>(SIZE_64 * SIZE_ONE_K);

        AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose{0, MMAD_BASE_SIZE, MMAD_BASE_SIZE, 0, 0, false,
                                                                   0};
        AscendC::MmadParams commonMadParams{MMAD_BASE_SIZE, MMAD_BASE_SIZE, MMAD_BASE_SIZE, 3, false, true};
        AscendC::FixpipeParamsV220 commonFixpipeParamsV220{MMAD_BASE_SIZE, MMAD_BASE_SIZE, MMAD_BASE_SIZE,
                                                           MMAD_BASE_SIZE, false};

        uint32_t ping_pong_flag_l0_a_ = 0;
        uint32_t ping_pong_flag_l0_b_ = 0;
        uint32_t ping_pong_flag_l0_c_ = 0;

        constexpr event_t eventIdMte1AToM = static_cast<event_t>(0);
        constexpr event_t eventIdMte1BToM = static_cast<event_t>(1);
        constexpr event_t eventIdMte1ToMte2 = static_cast<event_t>(0);

        AscendC::SetLoadDataPaddingValue(static_cast<ElementA>(0));
        // Arm M_MTE1{3..6} + FIX_M{0,1} before Cube1Compute (sameAB Aligned128 protocol)
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(6);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);

        // P8 dyv→qk tried 2026-11-06: precision PASS but 020 +0.1ms / 007 +0.7ms
        // vs P7 — rolled back. Keep qk→dyv (L1 drain between gemms).
        Cube1ComputeTscm(queryGm, keyGm, mm2WorkspaceGm, static_cast<int32_t>(d), dbParam, aL1Tensor, bL1Tensor,
                         l0_a_ping_tensor, l0_a_pong_tensor, l0_b_ping_tensor, l0_b_pong_tensor, l0_c_ping_tensor,
                         l0_c_pong_tensor, commonLoadData2dParamsNoTranspose, commonMadParams,
                         commonFixpipeParamsV220, ping_pong_flag_l0_a_, ping_pong_flag_l0_b_, ping_pong_flag_l0_c_,
                         eventIdMte1AToM, eventIdMte1BToM);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);

        Cube1ComputeTscm(dxGm, valueGm, mm1WorkspaceGm, static_cast<int32_t>(value_d), dbParam, aL1Tensor, bL1Tensor,
                         l0_a_ping_tensor, l0_a_pong_tensor, l0_b_ping_tensor, l0_b_pong_tensor, l0_c_ping_tensor,
                         l0_c_pong_tensor, commonLoadData2dParamsNoTranspose, commonMadParams,
                         commonFixpipeParamsV220, ping_pong_flag_l0_a_, ping_pong_flag_l0_b_, ping_pong_flag_l0_c_,
                         eventIdMte1AToM, eventIdMte1BToM);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);

        (void)L1_B_BYTES;
    }

#endif // FAG_COMMON_CUBE1_COMPUTE_TSCM_HPP
