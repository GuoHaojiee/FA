#!/usr/bin/env python3
"""
分析 CAModel trace_core0.json 提取真实 NPU 性能数据
"""
import json
import sys
from collections import defaultdict

def analyze_trace(trace_file):
    print(f"正在加载 trace 文件: {trace_file}")
    with open(trace_file, 'r') as f:
        data = json.load(f)

    # 统计各个计算单元的信息
    unit_stats = defaultdict(lambda: {
        'count': 0,
        'total_duration': 0,
        'min_ts': float('inf'),
        'max_te': 0
    })

    # 收集所有事件
    events = []
    for event in data:
        if event.get('ph') == 'X':  # Duration events
            events.append(event)

            # 提取单元名称
            if 'args' in event and 'name' in event['args']:
                unit_name = event['args']['name']
            else:
                unit_name = event.get('name', 'Unknown')

            # 统计时长
            dur = event.get('dur', 0)
            ts = event.get('ts', 0)

            unit_stats[unit_name]['count'] += 1
            unit_stats[unit_name]['total_duration'] += dur
            unit_stats[unit_name]['min_ts'] = min(unit_stats[unit_name]['min_ts'], ts)
            unit_stats[unit_name]['max_te'] = max(unit_stats[unit_name]['max_te'], ts + dur)

    # 计算总执行周期
    if events:
        all_ts = [e.get('ts', 0) for e in events if 'ts' in e]
        all_te = [e.get('ts', 0) + e.get('dur', 0) for e in events if 'ts' in e and 'dur' in e]

        if all_ts and all_te:
            total_start = min(all_ts)
            total_end = max(all_te)
            total_cycles = total_end - total_start

            print("\n" + "="*80)
            print("仿真性能分析结果")
            print("="*80)

            print(f"\n【关键性能指标】")
            print(f"  总执行周期数 (Cycles): {total_cycles:,}")
            print(f"  总事件数: {len(events):,}")

            # 假设时钟频率为 1 GHz (1000 MHz)，计算理论执行时间
            freq_mhz = 1000  # Ascend950 典型频率
            exec_time_us = total_cycles / freq_mhz
            exec_time_ms = exec_time_us / 1000

            print(f"\n【理论 NPU 执行时间】(假设 {freq_mhz} MHz 时钟)")
            print(f"  执行时间: {exec_time_ms:.3f} ms ({exec_time_us:.1f} μs)")
            print(f"  性能: {total_cycles / exec_time_ms:.0f} cycles/ms")

            print(f"\n【与仿真软件时间对比】")
            print(f"  CAModel 仿真时间: 204.13 秒 (204130 ms)")
            print(f"  真实 NPU 执行时间: {exec_time_ms:.3f} ms")
            print(f"  软件开销倍数: {204130 / exec_time_ms:.0f}x")
            print(f"  => CAModel 是软件仿真，运行慢是正常的")
            print(f"  => 真实 NPU 硬件执行只需 {exec_time_ms:.3f} ms!")

    # 分析各计算单元
    print(f"\n" + "="*80)
    print("各计算单元执行统计")
    print("="*80)

    # 按总时长排序
    sorted_units = sorted(unit_stats.items(),
                         key=lambda x: x[1]['total_duration'],
                         reverse=True)

    print(f"\n{'单元名称':<30} {'指令数':>10} {'总周期':>15} {'平均周期':>12} {'占比':>8}")
    print("-" * 80)

    for unit_name, stats in sorted_units[:20]:  # 显示前20个单元
        avg_dur = stats['total_duration'] / stats['count'] if stats['count'] > 0 else 0
        pct = (stats['total_duration'] / total_cycles * 100) if total_cycles > 0 else 0
        print(f"{unit_name:<30} {stats['count']:>10,} {stats['total_duration']:>15,} {avg_dur:>12.1f} {pct:>7.2f}%")

    # 按类别汇总
    print(f"\n" + "="*80)
    print("按功能单元类别汇总")
    print("="*80)

    category_stats = defaultdict(lambda: {'count': 0, 'duration': 0})

    for unit_name, stats in unit_stats.items():
        # 分类计算单元
        if 'VEC' in unit_name or 'VECTOR' in unit_name:
            category = 'VECTOR (向量运算)'
        elif 'CUBE' in unit_name or 'Cube' in unit_name:
            category = 'CUBE (矩阵乘)'
        elif 'SCALAR' in unit_name:
            category = 'SCALAR (标量运算)'
        elif 'MTE1' in unit_name:
            category = 'MTE1 (L1→L0 搬运)'
        elif 'MTE2' in unit_name:
            category = 'MTE2 (DDR/L2→L1/L0 搬运)'
        elif 'MTE3' in unit_name:
            category = 'MTE3 (UBUF→DDR/L2/L1 搬运)'
        elif 'FIXP' in unit_name:
            category = 'FIXP (L0C→OUT 搬运)'
        elif 'FLOW' in unit_name:
            category = 'FLOWCTRL (控制流)'
        else:
            category = 'OTHER (其他)'

        category_stats[category]['count'] += stats['count']
        category_stats[category]['duration'] += stats['total_duration']

    sorted_categories = sorted(category_stats.items(),
                              key=lambda x: x[1]['duration'],
                              reverse=True)

    print(f"\n{'功能单元':<30} {'指令数':>10} {'总周期':>15} {'占比':>8}")
    print("-" * 80)

    for category, stats in sorted_categories:
        pct = (stats['duration'] / total_cycles * 100) if total_cycles > 0 else 0
        print(f"{category:<30} {stats['count']:>10,} {stats['duration']:>15,} {pct:>7.2f}%")

    print(f"\n" + "="*80)

    return {
        'total_cycles': total_cycles,
        'total_events': len(events),
        'exec_time_ms': exec_time_ms if 'exec_time_ms' in locals() else 0,
        'unit_stats': dict(unit_stats),
        'category_stats': dict(category_stats)
    }

if __name__ == '__main__':
    trace_file = '/home/guohaojie/Guo/FA算子/ops-transformer/attention/flash_attention_score/examples/build/bin/sim_output/cannsim_20260624121042_test_aclnn_flash_attention_score_v2_fp32/report/trace_core0.json'

    if len(sys.argv) > 1:
        trace_file = sys.argv[1]

    try:
        result = analyze_trace(trace_file)

        print("\n✅ 分析完成！")
        print(f"\n关键结论:")
        print(f"  - 真实 NPU 执行周期: {result['total_cycles']:,} cycles")
        print(f"  - 真实 NPU 执行时间: {result['exec_time_ms']:.3f} ms (@ 1 GHz)")
        print(f"  - CAModel 仿真时间: 204130 ms (软件开销)")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {trace_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON 解析失败 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
