/**
 * Stub implementation for aclnnFlashAttentionScoreV2
 * This allows compilation without the full FA operator library
 * The actual execution will be handled by the CAModel simulator
 */

#include "aclnn/aclnn_base.h"
#include "aclnn_flash_attention_score.h"

extern "C" {

// Stub implementation for GetWorkspaceSize
aclnnStatus aclnnFlashAttentionScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double scaleValue,
    double keepProb,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,
    int64_t pseType,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // Stub implementation - set a default workspace size
    *workspaceSize = 1024 * 1024 * 10;  // 10MB default
    *executor = reinterpret_cast<aclOpExecutor*>(0x1);  // Dummy executor pointer
    return ACL_SUCCESS;
}

// Stub implementation for FlashAttentionScoreV2
aclnnStatus aclnnFlashAttentionScoreV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
{
    // Stub implementation - the actual execution will be done by simulator
    return ACL_SUCCESS;
}

// Stub for DlogRecordInner to resolve linking error
void DlogRecordInner(int module, int level, const char* fmt, ...)
{
    // Stub implementation - no-op
}

}  // extern "C"
