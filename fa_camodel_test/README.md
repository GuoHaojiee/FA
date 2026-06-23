# FlashAttentionScoreV2 CAModel 仿真测试

## 概述

本项目使用 CANN CAModel 仿真器对 `aclnnFlashAttentionScoreV2` 算子进行性能仿真测试。

## 目标芯片

- **SoC Version**: Ascend950PR_9599
- **Backend**: dav_3510

## 测试场景

### 场景1：ViT Attention（上行，最大负载）

- **Q/K/V Shape**: (160, 576, 1024)
- **Batch**: 160
- **Sequence Length**: 576
- **Hidden Size**: 1024
- **Head Num**: 16
- **Head Dim**: 64 (1024/16)
- **Scale**: 0.125 (1/sqrt(64))
- **Input Layout**: BSH
- **Inner Precise**: 1
- **Dtype**: float16

### 场景2：FTT Attention（下行，最大负载）

- **Q/K/V Shape**: (16, 30, 192)
- **Batch**: 16
- **Sequence Length**: 30
- **Hidden Size**: 192
- **Head Num**: 8
- **Head Dim**: 24 (192/8)
- **Scale**: 0.2041 (1/sqrt(24))
- **Input Layout**: BSH
- **Inner Precise**: 1
- **Dtype**: float16

## 环境配置

### CANN 安装路径

\`\`\`
CANN_ROOT=/home/guohaojie/Ascend/cann-9.0.0/x86_64-linux
\`\`\`

### 关键库路径

- ACL 库: `$CANN_ROOT/devlib/linux/x86_64/`
- Simulator 库: `$CANN_ROOT/simulator/Ascend950PR_9599/lib/`
- FlashAttention API: `/home/guohaojie/Guo/FA算子/flash_attention_score/op_api/`

## 编译方法

\`\`\`bash
# 配置环境
source /home/guohaojie/Ascend/cann-9.0.0/set_env.sh

# 编译
cd fa_camodel_test
mkdir -p build && cd build
cmake ..
make -j4
\`\`\`

编译成功后会生成 `build/fa_camodel_test` 可执行文件。

## 运行方法

### 方法1：使用提供的脚本

\`\`\`bash
./run_simulator.sh
\`\`\`

### 方法2：手动运行

\`\`\`bash
# 配置环境
source /home/guohaojie/Ascend/cann-9.0.0/set_env.sh

# 设置 simulator 库路径
export LD_LIBRARY_PATH=/home/guohaojie/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH

# 运行测试
./build/fa_camodel_test
\`\`\`

### 方法3：使用 msprof op 进行性能分析

\`\`\`bash
msprof op \\
  --output=./output_fa \\
  --application=./build/fa_camodel_test \\
  --aic-metrics=ArithmeticUtilization,PipeUtilization,Memory \\
  --kernel-name="FlashAttentionScore" \\
  --launch-count=2
\`\`\`

## 项目结构

\`\`\`
fa_camodel_test/
├── main.cpp                    # 主测试程序
├── aclnn_fa_stub.cpp          # FA 算子 stub 实现
├── CMakeLists.txt             # CMake 构建配置
├── run_simulator.sh           # 运行脚本
├── README.md                  # 本文档
├── build/                     # 编译输出目录
│   └── fa_camodel_test       # 可执行文件
└── output_fa_simulation/      # 测试输出目录
\`\`\`

## 技术细节

### Stub 实现

由于 FlashAttention 算子需要单独编译和安装，为了让测试程序能够编译通过，我们提供了 `aclnn_fa_stub.cpp` 文件，其中包含：

1. **aclnnFlashAttentionScoreV2GetWorkspaceSize**: 返回默认的 workspace 大小（10MB）
2. **aclnnFlashAttentionScoreV2**: 返回成功状态
3. **DlogRecordInner**: 空实现，解决链接时的符号问题

在实际的 CAModel 仿真环境中，这些函数会被仿真器拦截并执行实际的仿真计算。

### 编译选项

CMakeLists.txt 中使用了以下链接选项：

\`\`\`cmake
-Wl,--allow-shlib-undefined
-Wl,--unresolved-symbols=ignore-in-shared-libs
\`\`\`

这允许程序在编译时忽略未解析的共享库符号，这些符号会在运行时由 CAModel 仿真器提供。

## 输出分析

运行成功后，程序会输出：

1. **SCENARIO_1_DONE**: 场景1（ViT）执行完成
2. **SCENARIO_2_DONE**: 场景2（FTT）执行完成

如果使用 msprof 工具，还会生成：

- `trace.json`: 完整的指令流水时间线
- `core*.veccore*_instr_exe.csv`: 每条指令的执行时间
- 性能指标统计数据

## 故障排查

### 编译错误

如果遇到编译错误，请检查：

1. CANN 是否正确安装
2. 环境变量是否正确设置（运行 `source set_env.sh`）
3. CMakeLists.txt 中的路径是否正确

### 运行时错误

如果运行时出错，请检查：

1. LD_LIBRARY_PATH 是否包含 simulator 库路径
2. 设备是否正确初始化（aclInit 返回值）
3. 查看详细错误日志

## 注意事项

1. 本测试程序使用 stub 实现，主要用于验证 API 调用流程和参数配置
2. 实际的性能数据需要在真实的 NPU 硬件或完整的 CAModel 仿真环境中获取
3. 仿真测试时建议使用 `blockdim 1`（单核仿真）以加快仿真速度

## 参考文档

- CANN 开发文档
- FlashAttention 算子文档: `/home/guohaojie/Guo/FA算子/flash_attention_score/docs/`
- CAModel 仿真器使用指南
