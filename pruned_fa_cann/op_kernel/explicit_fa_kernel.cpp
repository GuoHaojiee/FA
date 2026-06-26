/**
 * Explicit FA kernel instantiation for OPP binary replacement.
 *
 * Tiling key 2328361008425337360 (0x2050000040100210) decodes to:
 *   KernelTypeKey=0, ImplMode=0, Layout=1 (LAYOUT_BSH), S1=64, S2=128,
 *   DTemplateType=64, DvTemplateType=0, PseMode=9 (PSE_NONE_TYPE),
 *   HasAtten=false, HasDrop=false, HasRope=false, OutDtype=0, Regbase=1
 *
 * Test case: B=2 S=30 H=192 (headNum=8, headDim=24), fp16, BSH, no optional tensors.
 * D=64 = template block size ≥ headDim=24. Dv=0 → same as D.
 *
 * Produces exactly:
 *   FlashAttentionScore_236874ba12878eb501b11f3f4feb2ce4_2328361008425337360_mix_aic (cube)
 *   FlashAttentionScore_236874ba12878eb501b11f3f4feb2ce4_2328361008425337360_mix_aiv (vec)
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include <flash_attention_score_tiling_regbase.h>
using namespace optiling;
#include "arch35/flash_attention_score_entry_regbase.h"
using namespace AscendC;

// ImplMode=0, Layout=1(BSH), S1=64, S2=128, D=64, Dv=0, PseMode=9(PSE_NONE),
// HasAtten=false, HasDrop=false, HasRope=false, OutDtype=0, Regbase=1
#define FA_EXPLICIT_PARAMS 0, 1, 64, 128, 64, 0, 9, false, false, false, 0, 1

#ifdef __DAV_C310_CUBE__

extern "C" __global__ __aicore__ void
FlashAttentionScore_236874ba12878eb501b11f3f4feb2ce4_2328361008425337360_mix_aic(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *deqScaleQ,
    __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope, __gm__ uint8_t *sink, __gm__ uint8_t *pScale,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttentionScoreSimplifiedTilingData);
    flash_attention_score_regbase<FA_EXPLICIT_PARAMS>(
        query, key, value, pse, dropMask, paddingMask, attenMask, prefix,
        actualSeqLengths, actualSeqLengthsKv, qStartIdx, kvStartIdx,
        deqScaleQ, deqScaleK, deqScaleV, pScale, queryRope, keyRope,
        softmaxMax, softmaxSum, softmaxOut, attentionOut, workspace, tiling);
}

#else  // vector core

extern "C" __global__ __aicore__ void
FlashAttentionScore_236874ba12878eb501b11f3f4feb2ce4_2328361008425337360_mix_aiv(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *deqScaleQ,
    __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope, __gm__ uint8_t *sink, __gm__ uint8_t *pScale,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttentionScoreSimplifiedTilingData);
    flash_attention_score_regbase<FA_EXPLICIT_PARAMS>(
        query, key, value, pse, dropMask, paddingMask, attenMask, prefix,
        actualSeqLengths, actualSeqLengthsKv, qStartIdx, kvStartIdx,
        deqScaleQ, deqScaleK, deqScaleV, pScale, queryRope, keyRope,
        softmaxMax, softmaxSum, softmaxOut, attentionOut, workspace, tiling);
}

#endif  // __DAV_C310_CUBE__
