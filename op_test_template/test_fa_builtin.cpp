/**
 * CAModel Simulator Test for aclnnFlashAttentionScoreV2
 * Test scenarios:
 * 1. ViT Attention (160, 576, 1024) - Uplink Max Load
 * 2. FTT Attention (16, 30, 192) - Downlink Max Load
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void CleanupTensors(aclTensor* q, aclTensor* k, aclTensor* v, aclTensor* softmaxMax,
                   aclTensor* softmaxSum, aclTensor* attentionOut) {
  if (q) aclDestroyTensor(q);
  if (k) aclDestroyTensor(k);
  if (v) aclDestroyTensor(v);
  if (softmaxMax) aclDestroyTensor(softmaxMax);
  if (softmaxSum) aclDestroyTensor(softmaxSum);
  if (attentionOut) aclDestroyTensor(attentionOut);
}

void CleanupMemory(void* qAddr, void* kAddr, void* vAddr, void* softmaxMaxAddr,
                   void* softmaxSumAddr, void* attentionOutAddr, void* workspaceAddr) {
  if (qAddr) aclrtFree(qAddr);
  if (kAddr) aclrtFree(kAddr);
  if (vAddr) aclrtFree(vAddr);
  if (softmaxMaxAddr) aclrtFree(softmaxMaxAddr);
  if (softmaxSumAddr) aclrtFree(softmaxSumAddr);
  if (attentionOutAddr) aclrtFree(attentionOutAddr);
  if (workspaceAddr) aclrtFree(workspaceAddr);
}

int TestScenario1_ViT(aclrtStream stream) {
  LOG_PRINT("\n========== SCENARIO 1: ViT Attention ==========\n");
  LOG_PRINT("Shape: (160, 576, 1024), head_num=16, head_dim=64\n");
  LOG_PRINT("Scale: 0.125, input_layout: BSH\n\n");

  // Scenario parameters
  int64_t B = 160;
  int64_t S = 576;
  int64_t H = 1024;
  int64_t headNum = 16;
  int64_t headDim = 64;
  double scaleValue = 0.125;  // 1/sqrt(64)

  // Shape definitions
  std::vector<int64_t> qShape = {B, S, H};
  std::vector<int64_t> kShape = {B, S, H};
  std::vector<int64_t> vShape = {B, S, H};
  std::vector<int64_t> softmaxMaxShape = {B, headNum, S, 8};
  std::vector<int64_t> softmaxSumShape = {B, headNum, S, 8};
  std::vector<int64_t> attentionOutShape = {B, S, H};

  // Device pointers
  void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
  void *softmaxMaxAddr = nullptr, *softmaxSumAddr = nullptr;
  void *attentionOutAddr = nullptr, *workspaceAddr = nullptr;

  // Tensor pointers
  aclTensor *q = nullptr, *k = nullptr, *v = nullptr;
  aclTensor *softmaxMax = nullptr, *softmaxSum = nullptr;
  aclTensor *attentionOut = nullptr;

  // Create host data (using float16, but allocating as uint16_t)
  int64_t qkvSize = GetShapeSize(qShape);
  int64_t softmaxSize = GetShapeSize(softmaxMaxShape);

  std::vector<uint16_t> qHostData(qkvSize, 0x3C00);  // fp16(1.0)
  std::vector<uint16_t> kHostData(qkvSize, 0x3C00);
  std::vector<uint16_t> vHostData(qkvSize, 0x3C00);
  std::vector<float> softmaxMaxHostData(softmaxSize, 0.0f);
  std::vector<float> softmaxSumHostData(softmaxSize, 0.0f);
  std::vector<uint16_t> attentionOutHostData(qkvSize, 0);

  auto ret = CreateAclTensor(qHostData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(kHostData, kShape, &kAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(vHostData, vShape, &vAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutAddr, aclDataType::ACL_FLOAT16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  // Operator parameters
  double keepProb = 1.0;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t innerPrecise = 1;
  int64_t sparseMode = 0;
  int64_t pseType = 0;
  char inputLayout[] = "BSH";

  // Call aclnnFlashAttentionScoreV2GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(
            q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout,
            innerPrecise, sparseMode, pseType,
            softmaxMax, softmaxSum, nullptr, attentionOut,
            &workspaceSize, &executor);

  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnFlashAttentionScoreV2GetWorkspaceSize failed. ERROR: %d\n", ret);
    LOG_PRINT("[ERROR msg] %s\n", aclGetRecentErrMsg());
    CleanupTensors(q, k, v, softmaxMax, softmaxSum, attentionOut);
    CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr);
    return ret;
  }

  LOG_PRINT("Workspace size: %lu bytes\n", workspaceSize);

  // Allocate workspace
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              CleanupTensors(q, k, v, softmaxMax, softmaxSum, attentionOut);
              CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr);
              return ret);
  }

  // Execute operator
  ret = aclnnFlashAttentionScoreV2(workspaceAddr, workspaceSize, executor, stream);

  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnFlashAttentionScoreV2 failed. ERROR: %d\n", ret);
    LOG_PRINT("[ERROR msg] %s\n", aclGetRecentErrMsg());
  } else {
    LOG_PRINT("Operator executed successfully!\n");
  }

  // Synchronize
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Cleanup
  CleanupTensors(q, k, v, softmaxMax, softmaxSum, attentionOut);
  CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr);

  LOG_PRINT("SCENARIO_1_DONE\n");
  return 0;
}

int TestScenario2_FTT(aclrtStream stream) {
  LOG_PRINT("\n========== SCENARIO 2: FTT Attention ==========\n");
  LOG_PRINT("Shape: (16, 30, 192), head_num=8, head_dim=24\n");
  LOG_PRINT("Scale: 0.2041, input_layout: BSH\n\n");

  // Scenario parameters
  int64_t B = 2;
  int64_t S = 30;
  int64_t H = 192;
  int64_t headNum = 8;
  int64_t headDim = 24;
  double scaleValue = 0.2041;  // 1/sqrt(24)

  // Shape definitions
  std::vector<int64_t> qShape = {B, S, H};
  std::vector<int64_t> kShape = {B, S, H};
  std::vector<int64_t> vShape = {B, S, H};
  std::vector<int64_t> softmaxMaxShape = {B, headNum, S, 8};
  std::vector<int64_t> softmaxSumShape = {B, headNum, S, 8};
  std::vector<int64_t> attentionOutShape = {B, S, H};

  // Device pointers
  void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
  void *softmaxMaxAddr = nullptr, *softmaxSumAddr = nullptr;
  void *attentionOutAddr = nullptr, *workspaceAddr = nullptr;

  // Tensor pointers
  aclTensor *q = nullptr, *k = nullptr, *v = nullptr;
  aclTensor *softmaxMax = nullptr, *softmaxSum = nullptr;
  aclTensor *attentionOut = nullptr;

  // Create host data
  int64_t qkvSize = GetShapeSize(qShape);
  int64_t softmaxSize = GetShapeSize(softmaxMaxShape);

  std::vector<uint16_t> qHostData(qkvSize, 0x3C00);  // fp16(1.0)
  std::vector<uint16_t> kHostData(qkvSize, 0x3C00);
  std::vector<uint16_t> vHostData(qkvSize, 0x3C00);
  std::vector<float> softmaxMaxHostData(softmaxSize, 0.0f);
  std::vector<float> softmaxSumHostData(softmaxSize, 0.0f);
  std::vector<uint16_t> attentionOutHostData(qkvSize, 0);

  auto ret = CreateAclTensor(qHostData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(kHostData, kShape, &kAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(vHostData, vShape, &vAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutAddr, aclDataType::ACL_FLOAT16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr); return ret);

  // Operator parameters
  double keepProb = 1.0;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t innerPrecise = 1;
  int64_t sparseMode = 0;
  int64_t pseType = 0;
  char inputLayout[] = "BSH";

  // Call aclnnFlashAttentionScoreV2GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(
            q, k, v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout,
            innerPrecise, sparseMode, pseType,
            softmaxMax, softmaxSum, nullptr, attentionOut,
            &workspaceSize, &executor);

  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnFlashAttentionScoreV2GetWorkspaceSize failed. ERROR: %d\n", ret);
    LOG_PRINT("[ERROR msg] %s\n", aclGetRecentErrMsg());
    CleanupTensors(q, k, v, softmaxMax, softmaxSum, attentionOut);
    CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr);
    return ret;
  }

  LOG_PRINT("Workspace size: %lu bytes\n", workspaceSize);

  // Allocate workspace
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              CleanupTensors(q, k, v, softmaxMax, softmaxSum, attentionOut);
              CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr);
              return ret);
  }

  // Execute operator
  ret = aclnnFlashAttentionScoreV2(workspaceAddr, workspaceSize, executor, stream);

  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnFlashAttentionScoreV2 failed. ERROR: %d\n", ret);
    LOG_PRINT("[ERROR msg] %s\n", aclGetRecentErrMsg());
  } else {
    LOG_PRINT("Operator executed successfully!\n");
  }

  // Synchronize
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Cleanup
  CleanupTensors(q, k, v, softmaxMax, softmaxSum, attentionOut);
  CleanupMemory(qAddr, kAddr, vAddr, softmaxMaxAddr, softmaxSumAddr, attentionOutAddr, workspaceAddr);

  LOG_PRINT("SCENARIO_2_DONE\n");
  return 0;
}

int main() {
  LOG_PRINT("========================================\n");
  LOG_PRINT("CANN CAModel Simulator Test\n");
  LOG_PRINT("aclnnFlashAttentionScoreV2\n");
  LOG_PRINT("Target SoC: Ascend950PR_9599\n");
  LOG_PRINT("========================================\n");

  // Initialize device and stream
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init ACL failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("ACL initialized successfully\n");
  LOG_PRINT("Device set to %d\n", deviceId);

  // Run test scenarios
  // TestScenario1_ViT(stream);  // skip ViT — too much CAModel memory

  ret = TestScenario2_FTT(stream);
  if (ret != 0) {
    LOG_PRINT("Scenario 2 failed\n");
  }

  // Cleanup
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("\n========================================\n");
  LOG_PRINT("All tests completed!\n");
  LOG_PRINT("========================================\n");

  return 0;
}
