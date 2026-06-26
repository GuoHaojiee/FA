/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * PRUNED: KernelTypeKey==1 (empty tensor) block removed — always dead for fp16/BSH params.
 *         empty_tensor_regbase.h include removed (was only for the dead block).
 *         Source: CANN 9.0.0 flash_attention_score_apt.cpp (no optionalDn parameter).
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include <flash_attention_score_tiling_regbase.h>
using namespace optiling;
#include "arch35/flash_attention_score_template_tiling_key.h"
#include "arch35/flash_attention_score_entry_regbase.h"
using namespace optiling;
using namespace AscendC;


template<uint8_t KernelTypeKey, uint8_t implMode, uint8_t layout, uint16_t s1TemplateType, uint16_t s2TemplateType,
    uint16_t dTemplateType, uint16_t dvTemplateType, uint8_t pseMode, bool hasAtten, bool hasDrop, bool hasRope,
    uint8_t outDtype, uint8_t regbase>
__global__ __aicore__ void
flash_attention_score(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                      __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                      __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                      __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *deqScaleQ,
                      __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *queryRope,
                      __gm__ uint8_t *keyRope, __gm__ uint8_t *sink, __gm__ uint8_t *pScale,
                      __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                      __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttentionScoreSimplifiedTilingData);
    // KernelTypeKey==1 block removed: always 0 (non-empty tensor) for fp16/BSH fixed params.
    flash_attention_score_regbase<implMode, layout, s1TemplateType, s2TemplateType, dTemplateType, dvTemplateType,
        pseMode, hasAtten, hasDrop, hasRope, outDtype, regbase>(query, key, value, pse, dropMask,
        paddingMask, attenMask, prefix, actualSeqLengths, actualSeqLengthsKv, qStartIdx, kvStartIdx, deqScaleQ,
        deqScaleK, deqScaleV, pScale, queryRope, keyRope, softmaxMax, softmaxSum, softmaxOut, attentionOut,
        workspace, tiling);
}
