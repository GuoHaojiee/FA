/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_apt.cpp
 * \brief
 *
 * PRUNING NOTE (pruned_fa round-1, 2026-06-26):
 *   Fixed params: soc=Ascend950PR_9599, dtype=fp16, layout=BSH,
 *                 B=2 S=30 H=192 headNum=8 headDim=24, no optional tensors.
 *   KernelTypeKey is always 0 (non-empty-tensor) for these params.
 *   → if constexpr (KernelTypeKey == 1) branch flattened away (was DEAD).
 *   → #include "flash_attention_score_empty_tensor_regbase.h" removed (only used by that branch).
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "arch35/flash_attention_score_template_tiling_key.h"
#include "arch35/flash_attention_score_entry_regbase.h"
#if __has_include("../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h")
#include "../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#else
#include "../common/arch35/flash_attention_score_tiling_regbase.h"
#endif
using namespace optiling;
using namespace AscendC;


template<uint8_t KernelTypeKey, uint8_t implMode, uint8_t layout, uint16_t s1TemplateType, uint16_t s2TemplateType,
    uint16_t dTemplateType, uint16_t dvTemplateType, uint8_t pseMode, bool hasAtten, bool hasDrop, bool hasRope,
    uint8_t outDtype, uint8_t regbase, bool optionalDn>
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
    // KernelTypeKey == 1 (empty-tensor) branch removed: always 0 for fp16/BSH fixed params.
    flash_attention_score_regbase<implMode, layout, s1TemplateType, s2TemplateType, dTemplateType, dvTemplateType,
        pseMode, hasAtten, hasDrop, hasRope, outDtype, regbase, optionalDn>(query, key, value, pse, dropMask,
        paddingMask, attenMask, prefix, actualSeqLengths, actualSeqLengthsKv, qStartIdx, kvStartIdx, deqScaleQ,
        deqScaleK, deqScaleV, pScale, queryRope, keyRope, softmaxMax, softmaxSum, softmaxOut, attentionOut, sink,
        workspace, tiling);
}
