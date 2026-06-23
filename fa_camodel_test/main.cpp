/**
 * CANN CAModel Simulator Test for aclnnFlashAttentionScoreV2
 * Target SoC: Ascend950PR_9599
 */

#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_flash_attention_score.h"

using namespace std;

// Helper function to create aclTensor
aclTensor* CreateTensor(const vector<int64_t>& dims, aclDataType dtype, aclFormat format, void* devPtr) {
    int64_t* dimArray = new int64_t[dims.size()];
    for (size_t i = 0; i < dims.size(); i++) {
        dimArray[i] = dims[i];
    }

    aclTensor* tensor = aclCreateTensor(dimArray, dims.size(), dtype, nullptr, 0,
                                         format, nullptr, 0, devPtr);
    delete[] dimArray;
    return tensor;
}

// Test scenario 1: ViT attention
void TestScenario1_ViT() {
    cout << "========== SCENARIO 1: ViT Attention ==========" << endl;
    cout << "Shape: (160, 576, 1024), head_num=16, head_dim=64" << endl;

    const int64_t batch = 160;
    const int64_t seqLen = 576;
    const int64_t hiddenSize = 1024;
    const int64_t headNum = 16;
    const int64_t headDim = 64;  // 1024 / 16
    const double scale = 0.125;  // 1/sqrt(64)
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    char inputLayout[] = "BSH";
    const int64_t innerPrecise = 1;
    const int64_t sparseMode = 0;
    const int64_t pseType = 0;

    // Q/K/V shape: (batch, seqLen, hiddenSize)
    int64_t qkvSize = batch * seqLen * hiddenSize * sizeof(aclFloat16);
    void *devQ, *devK, *devV;
    aclrtMalloc(&devQ, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devK, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devV, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclTensor* tensorQ = CreateTensor({batch, seqLen, hiddenSize}, ACL_FLOAT16, ACL_FORMAT_ND, devQ);
    aclTensor* tensorK = CreateTensor({batch, seqLen, hiddenSize}, ACL_FLOAT16, ACL_FORMAT_ND, devK);
    aclTensor* tensorV = CreateTensor({batch, seqLen, hiddenSize}, ACL_FLOAT16, ACL_FORMAT_ND, devV);

    // softmaxMax/Sum shape: (batch, headNum, seqLen, 8)
    int64_t softmaxSize = batch * headNum * seqLen * 8 * sizeof(float);
    void *devSoftmaxMax, *devSoftmaxSum;
    aclrtMalloc(&devSoftmaxMax, softmaxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSoftmaxSum, softmaxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclTensor* tensorSoftmaxMax = CreateTensor({batch, headNum, seqLen, 8}, ACL_FLOAT, ACL_FORMAT_ND, devSoftmaxMax);
    aclTensor* tensorSoftmaxSum = CreateTensor({batch, headNum, seqLen, 8}, ACL_FLOAT, ACL_FORMAT_ND, devSoftmaxSum);

    // attentionOut shape: same as Q
    void* devAttentionOut;
    aclrtMalloc(&devAttentionOut, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclTensor* tensorAttentionOut = CreateTensor({batch, seqLen, hiddenSize}, ACL_FLOAT16, ACL_FORMAT_ND, devAttentionOut);

    // Get workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(
        tensorQ, tensorK, tensorV,
        nullptr,  // realShiftOptional
        nullptr,  // dropMaskOptional
        nullptr,  // paddingMaskOptional
        nullptr,  // attenMaskOptional
        nullptr,  // prefixOptional
        nullptr,  // qStartIdxOptional
        nullptr,  // kvStartIdxOptional
        scale, keepProb, preTokens, nextTokens, headNum,
        inputLayout, innerPrecise, sparseMode, pseType,
        tensorSoftmaxMax, tensorSoftmaxSum,
        nullptr,  // softmaxOutOut
        tensorAttentionOut,
        &workspaceSize, &executor
    );

    if (ret != ACL_SUCCESS) {
        cout << "Failed to get workspace size: " << ret << endl;
        return;
    }
    cout << "Workspace size: " << workspaceSize << " bytes" << endl;

    // Allocate workspace
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // Execute operator
    aclrtStream stream;
    aclrtCreateStream(&stream);

    ret = aclnnFlashAttentionScoreV2(workspace, workspaceSize, executor, stream);

    if (ret != ACL_SUCCESS) {
        cout << "Failed to execute operator: " << ret << endl;
    } else {
        cout << "Operator executed successfully!" << endl;
    }

    aclrtSynchronizeStream(stream);

    // Cleanup
    if (workspace) aclrtFree(workspace);
    aclrtFree(devQ);
    aclrtFree(devK);
    aclrtFree(devV);
    aclrtFree(devSoftmaxMax);
    aclrtFree(devSoftmaxSum);
    aclrtFree(devAttentionOut);
    aclDestroyTensor(tensorQ);
    aclDestroyTensor(tensorK);
    aclDestroyTensor(tensorV);
    aclDestroyTensor(tensorSoftmaxMax);
    aclDestroyTensor(tensorSoftmaxSum);
    aclDestroyTensor(tensorAttentionOut);
    aclrtDestroyStream(stream);

    cout << "SCENARIO_1_DONE" << endl;
}

// Test scenario 2: FTT attention
void TestScenario2_FTT() {
    cout << "\n========== SCENARIO 2: FTT Attention ==========" << endl;
    cout << "Shape: (16, 30, 192), head_num=8, head_dim=24" << endl;

    const int64_t batch = 16;
    const int64_t seqLen = 30;
    const int64_t hiddenSize = 192;
    const int64_t headNum = 8;
    const int64_t headDim = 24;  // 192 / 8
    const double scale = 0.2041;  // 1/sqrt(24)
    const double keepProb = 1.0;
    const int64_t preTokens = 65536;
    const int64_t nextTokens = 65536;
    char inputLayout[] = "BSH";
    const int64_t innerPrecise = 1;
    const int64_t sparseMode = 0;
    const int64_t pseType = 0;

    // Q/K/V shape: (batch, seqLen, hiddenSize)
    int64_t qkvSize = batch * seqLen * hiddenSize * sizeof(aclFloat16);
    void *devQ, *devK, *devV;
    aclrtMalloc(&devQ, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devK, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devV, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclTensor* tensorQ = CreateTensor({batch, seqLen, hiddenSize}, ACL_FLOAT16, ACL_FORMAT_ND, devQ);
    aclTensor* tensorK = CreateTensor({batch, seqLen, hiddenSize}, ACL_FLOAT16, ACL_FORMAT_ND, devK);
    aclTensor* tensorV = CreateTensor({batch, seqLen, hiddenSize}, ACL_FLOAT16, ACL_FORMAT_ND, devV);

    // softmaxMax/Sum shape: (batch, headNum, seqLen, 8)
    int64_t softmaxSize = batch * headNum * seqLen * 8 * sizeof(float);
    void *devSoftmaxMax, *devSoftmaxSum;
    aclrtMalloc(&devSoftmaxMax, softmaxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSoftmaxSum, softmaxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclTensor* tensorSoftmaxMax = CreateTensor({batch, headNum, seqLen, 8}, ACL_FLOAT, ACL_FORMAT_ND, devSoftmaxMax);
    aclTensor* tensorSoftmaxSum = CreateTensor({batch, headNum, seqLen, 8}, ACL_FLOAT, ACL_FORMAT_ND, devSoftmaxSum);

    // attentionOut shape: same as Q
    void* devAttentionOut;
    aclrtMalloc(&devAttentionOut, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclTensor* tensorAttentionOut = CreateTensor({batch, seqLen, hiddenSize}, ACL_FLOAT16, ACL_FORMAT_ND, devAttentionOut);

    // Get workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(
        tensorQ, tensorK, tensorV,
        nullptr,  // realShiftOptional
        nullptr,  // dropMaskOptional
        nullptr,  // paddingMaskOptional
        nullptr,  // attenMaskOptional
        nullptr,  // prefixOptional
        nullptr,  // qStartIdxOptional
        nullptr,  // kvStartIdxOptional
        scale, keepProb, preTokens, nextTokens, headNum,
        inputLayout, innerPrecise, sparseMode, pseType,
        tensorSoftmaxMax, tensorSoftmaxSum,
        nullptr,  // softmaxOutOut
        tensorAttentionOut,
        &workspaceSize, &executor
    );

    if (ret != ACL_SUCCESS) {
        cout << "Failed to get workspace size: " << ret << endl;
        return;
    }
    cout << "Workspace size: " << workspaceSize << " bytes" << endl;

    // Allocate workspace
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // Execute operator
    aclrtStream stream;
    aclrtCreateStream(&stream);

    ret = aclnnFlashAttentionScoreV2(workspace, workspaceSize, executor, stream);

    if (ret != ACL_SUCCESS) {
        cout << "Failed to execute operator: " << ret << endl;
    } else {
        cout << "Operator executed successfully!" << endl;
    }

    aclrtSynchronizeStream(stream);

    // Cleanup
    if (workspace) aclrtFree(workspace);
    aclrtFree(devQ);
    aclrtFree(devK);
    aclrtFree(devV);
    aclrtFree(devSoftmaxMax);
    aclrtFree(devSoftmaxSum);
    aclrtFree(devAttentionOut);
    aclDestroyTensor(tensorQ);
    aclDestroyTensor(tensorK);
    aclDestroyTensor(tensorV);
    aclDestroyTensor(tensorSoftmaxMax);
    aclDestroyTensor(tensorSoftmaxSum);
    aclDestroyTensor(tensorAttentionOut);
    aclrtDestroyStream(stream);

    cout << "SCENARIO_2_DONE" << endl;
}

int main(int argc, char* argv[]) {
    cout << "CANN CAModel Simulator Test for FlashAttentionScoreV2" << endl;
    cout << "Target SoC: Ascend950PR_9599" << endl << endl;

    // Initialize ACL
    const char* configFile = nullptr;
    aclError ret = aclInit(configFile);
    if (ret != ACL_SUCCESS) {
        cout << "aclInit failed: " << ret << endl;
        return -1;
    }
    cout << "ACL initialized successfully" << endl;

    // Set device (device 0)
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        cout << "aclrtSetDevice failed: " << ret << endl;
        aclFinalize();
        return -1;
    }
    cout << "Device set to " << deviceId << endl << endl;

    // Run test scenarios
    TestScenario1_ViT();
    TestScenario2_FTT();

    // Finalize
    aclrtResetDevice(deviceId);
    aclFinalize();

    cout << "\nAll tests completed!" << endl;
    return 0;
}
