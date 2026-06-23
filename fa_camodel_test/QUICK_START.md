# Quick Start Guide - FlashAttention CAModel 仿真测试

## 一键运行

```bash
cd /home/guohaojie/Guo/FA原生算子性能分析/fa_camodel_test
./run_simulator.sh
```

## 手动运行（推荐）

### Step 1: 设置环境

```bash
cd /home/guohaojie/Guo/FA原生算子性能分析/fa_camodel_test
source /home/guohaojie/Ascend/cann-9.0.0/set_env.sh
chmod 750 .
```

### Step 2: 运行仿真

```bash
msprof op simulator \
  --soc-version=Ascend950PR_9599 \
  --output=./output_fa_test \
  --application=./build/fa_camodel_test \
  --aic-metrics=PipeUtilization \
  --launch-count=2 \
  --timeout=30
```

### Step 3: 查看结果

```bash
# 查看输出目录
ls -R ./output_fa_test/OPPROF_*/

# 查看测试日志（如果有）
cat ./output_fa_test/OPPROF_*/device0/*.log
```

## 重新编译

```bash
cd build
rm -rf *
cmake ..
make -j4
```

## 测试场景

✅ **场景1 - ViT Attention**: (160, 576, 1024), 16 heads, head_dim=64
✅ **场景2 - FTT Attention**: (16, 30, 192), 8 heads, head_dim=24

## 预期输出

```
CANN CAModel Simulator Test for FlashAttentionScoreV2
Target SoC: Ascend950PR_9599

ACL initialized successfully
Device set to 0

========== SCENARIO 1: ViT Attention ==========
Workspace size: 10485760 bytes
Operator executed successfully!
SCENARIO_1_DONE

========== SCENARIO 2: FTT Attention ==========
Workspace size: 10485760 bytes
Operator executed successfully!
SCENARIO_2_DONE

All tests completed!
```

## 常见问题

### Q1: 编译失败

**A**: 确保已经设置 CANN 环境变量
```bash
source /home/guohaojie/Ascend/cann-9.0.0/set_env.sh
```

### Q2: 权限错误

**A**: 修改目录权限
```bash
chmod 750 /home/guohaojie/Guo/FA原生算子性能分析/fa_camodel_test
```

### Q3: 仿真器找不到

**A**: 检查 soc-version 参数
```bash
ls /home/guohaojie/Ascend/cann-9.0.0/x86_64-linux/simulator/
```

## 文档索引

- **README.md**: 完整使用说明
- **TEST_RESULTS.md**: 测试结果报告
- **SUMMARY.md**: 项目总结（在上级目录）
- **QUICK_START.md**: 本文档

## 联系与支持

如有问题，请查阅：
1. CANN 开发文档
2. FA 算子文档: `/home/guohaojie/Guo/FA算子/flash_attention_score/docs/`
3. 本项目的 README.md
