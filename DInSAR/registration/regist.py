#!/usr/bin/env python3
"""
配准和重采样主脚本
功能：整合粗配准、精配准、偏移估计、质量评估和重采样流程
输入：主图像名称（无后缀）、辅图像名称（无后缀）、精配准格网大小
输出：重采样后的辅图像文件
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入必要的模块
from coarse_regist import coarse_register, auto_coarse_register, read_image
from fine_regist import fine_register
from offset_proc import estimate_offsets, assess_registration_quality
from resampler import resample_with_yaml

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='配准和重采样主脚本')
    parser.add_argument('master_name', help='主图像名称（无后缀）')
    parser.add_argument('slave_name', help='辅图像名称（无后缀）')
    parser.add_argument('--output-dir', default='.', help='输出目录')
    parser.add_argument('--window-size', type=int, default=256, help='粗配准窗口大小')
    parser.add_argument('--search-range', type=int, default=64, help='粗配准搜索范围')
    parser.add_argument('--correlation-threshold', type=float, default=30, help='相关系数阈值')
    parser.add_argument('--num-workers', type=int, default=8, help='并行处理线程数')
    parser.add_argument('--manual-coarse', action='store_true', help='关闭粗配准自动模式，使用 --window-size/--search-range')
    parser.add_argument('--esd', action='store_true', help='启用 Sentinel-1 TOPS ESD 残余方位向配准微调')
    parser.add_argument('--esd-output-json', default=None, help='ESD 结果 JSON 输出路径（默认 output-dir/esd_result.json）')
    parser.add_argument('--esd-min-overlap-lines-ml', type=int, default=16, help='严格版 ESD：最小 overlap 行数（ml网格）')
    parser.add_argument('--esd-min-pair-coh', type=float, default=0.08, help='严格版 ESD：最小 pair coherence')
    parser.add_argument('--esd-min-delta-fd-hz', type=float, default=200.0, help='严格版 ESD：最小 |delta Doppler| (Hz)')
    parser.add_argument('--esd-max-dev-px', type=float, default=0.08, help='ESD 鲁棒筛选最大偏差（像素）')
    parser.add_argument('--esd-apply-low-reliability', action='store_true', help='即便 ESD 结果 reliability=low 也强制应用')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建文件路径
    master_vrt = f"{args.master_name}.vrt"
    slave_vrt = f"{args.slave_name}.vrt"
    master_yaml = f"{args.master_name}.yaml"
    slave_yaml = f"{args.slave_name}.yaml"
    offset_file = str(output_dir / "offset_estimate.txt")
    
    # 检查输入文件是否存在
    if not os.path.exists(master_vrt):
        print(f"错误：主图像VRT文件不存在: {master_vrt}")
        return 1
    
    if not os.path.exists(slave_vrt):
        print(f"错误：辅图像VRT文件不存在: {slave_vrt}")
        return 1
    
    if not os.path.exists(master_yaml):
        print(f"错误：主图像YAML文件不存在: {master_yaml}")
        return 1
    
    if not os.path.exists(slave_yaml):
        print(f"错误：辅图像YAML文件不存在: {slave_yaml}")
        return 1
    
    print("=== GMTSAR DInSAR 配准和重采样流程 ===")
    print(f"主图像: {args.master_name}")
    print(f"辅图像: {args.slave_name}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 1. 粗配准
    print("1. 执行粗配准...")
    try:
        master = read_image(master_vrt)
        slave = read_image(slave_vrt)
        
        if master is None or slave is None:
            print("错误：无法读取图像文件")
            return 1
        
        coarse_result = coarse_register(
            master, slave,
            window_size=args.window_size,
            search_range=args.search_range,
            correlation_threshold=args.correlation_threshold,
            output_file=offset_file,
            auto=(not args.manual_coarse)
        )

        coarse_window = coarse_result.get('best_window_size', coarse_result.get('window_size', args.window_size))
        coarse_search = coarse_result.get('best_search_range', coarse_result.get('search_range', args.search_range))
        coarse_init = coarse_result.get('initial_offset', (coarse_result['azimuth_offset'], coarse_result['range_offset']))
        print(f"  粗配准完成: 方位向偏移={coarse_result['azimuth_offset']:.4f}, 距离向偏移={coarse_result['range_offset']:.4f}")
        print(f"  最佳窗口参数: window_size={coarse_window}, search_range={coarse_search}")
        print(f"  初始偏移: az={coarse_init[0]:.4f}, rg={coarse_init[1]:.4f}")
    except Exception as e:
        print(f"  粗配准失败: {e}")
        return 1
    
    print()
    
    # 2. 精配准
    print("2. 执行精配准...")
    try:
        fine_result = fine_register(
            master, slave,
            initial_offset=coarse_init,
            grid_spacing=coarse_search,
            num_workers=args.num_workers,
            coarse_offset_file=offset_file,
            correlation_threshold=args.correlation_threshold,
            output_file=offset_file
        )
        
        print(f"  精配准完成: 配准点数={fine_result['num_points']}")
    except Exception as e:
        print(f"  精配准失败: {e}")
        return 1
    
    print()
    
    # 3. 偏移估计
    print("3. 执行偏移估计...")
    try:
        offset_result = estimate_offsets(
            offsets=None,
            input_file=offset_file,
            output_file=offset_file
        )
        
        print(f"  偏移估计完成: 方位向RMS={offset_result['final_residuals']['azimuth_rms']:.4f}, 距离向RMS={offset_result['final_residuals']['range_rms']:.4f}")
    except Exception as e:
        print(f"  偏移估计失败: {e}")
        return 1
    
    print()
    
    # 4. 质量评估
    print("4. 执行质量评估...")
    try:
        quality_result = assess_registration_quality(from_file=True, offset_file=offset_file)
        
        print(f"  质量评估完成: 质量等级={quality_result['quality']}, 置信度={quality_result['confidence']:.3f}")
    except Exception as e:
        print(f"  质量评估失败: {e}")
        # 质量评估失败不影响后续流程，继续执行
    
    print()
    
    # 5. 重采样（第一遍）
    print("5. 执行重采样...")
    output_yaml = output_dir / f"{args.slave_name}_resamp.yaml"
    resamp_tiff = output_dir / f"{args.slave_name}_resamp.tiff"
    resamp_vrt = output_dir / f"{args.slave_name}_resamp.vrt"
    final_offset_file = offset_file
    try:
        ok = resample_with_yaml(
            master_yaml,
            slave_yaml,
            str(output_yaml),
            offset_file,
            num_workers=args.num_workers
        )
        if not ok:
            print("  重采样失败: 详见上方错误日志")
            return 1
        print("  第一遍重采样完成")
    except Exception as e:
        print(f"  重采样失败: {e}")
        return 1

    # 5.1 ESD 残余方位向微调（可选）
    if args.esd:
        print("5.1 执行 ESD 残余方位向配准...")
        esd_script = Path(__file__).parent / "esd_regist.py"
        esd_json = Path(args.esd_output_json) if args.esd_output_json else (output_dir / "esd_result.json")
        esd_offset_file = output_dir / "offset_estimate_esd.txt"
        try:
            cmd = [
                sys.executable,
                str(esd_script),
                master_vrt,
                str(resamp_vrt),
                "--master-yaml",
                master_yaml,
                "--slave-yaml",
                slave_yaml,
                "--min-overlap-lines-ml", str(args.esd_min_overlap_lines_ml),
                "--min-pair-coh", str(args.esd_min_pair_coh),
                "--min-delta-fd-hz", str(args.esd_min_delta_fd_hz),
                "--max-dev-px", str(args.esd_max_dev_px),
                "--output-json", str(esd_json),
                "--offset-file", str(offset_file),
                "--output-offset-file", str(esd_offset_file),
            ]
            print(f"  [RUN] {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=str(Path.cwd()))
        except Exception as e:
            print(f"  ESD 估计失败，保留原始配准结果: {e}")
        else:
            reliability = "unknown"
            try:
                with open(esd_json, "r", encoding="utf-8") as f:
                    esd_info = json.load(f)
                reliability = str(esd_info.get("reliability", "unknown")).lower()
                esd_shift = float(esd_info.get("esd_azimuth_offset_px", 0.0))
                esd_std = float(esd_info.get("esd_azimuth_std_px", float("nan")))
                print(f"  ESD 结果: shift={esd_shift:.6f} px, std={esd_std:.6f}, reliability={reliability}")
            except Exception as e:
                print(f"  警告: 读取 ESD JSON 失败，默认按未知可靠性处理: {e}")

            apply_esd = True
            if (reliability == "low") and (not args.esd_apply_low_reliability):
                apply_esd = False
                print("  ESD reliability=low，默认不应用；可用 --esd-apply-low-reliability 强制应用。")

            if apply_esd:
                try:
                    print("  应用 ESD 修正并重采样（第二遍）...")
                    ok2 = resample_with_yaml(
                        master_yaml,
                        slave_yaml,
                        str(output_yaml),
                        str(esd_offset_file),
                        num_workers=args.num_workers
                    )
                    if not ok2:
                        print("  ESD 修正后的重采样失败，保留第一遍结果")
                    else:
                        final_offset_file = str(esd_offset_file)
                        print(f"  ESD 修正已应用: {esd_offset_file}")
                except Exception as e:
                    print(f"  ESD 修正重采样失败，保留第一遍结果: {e}")

    print(f"  重采样完成")
    print(f"  输出文件:")
    print(f"    - {resamp_tiff}")
    print(f"    - {resamp_vrt}")
    print(f"    - {output_yaml}")
    print(f"  使用偏移模型: {final_offset_file}")
    
    print()
    print("=== 配准和重采样流程完成 ===")
    return 0

if __name__ == '__main__':
    sys.exit(main())
