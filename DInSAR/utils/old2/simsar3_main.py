


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SAR数据模拟器 V3 - 从DEM生成模拟SAR图像',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('yaml_file', help='SAR参数YAML文件')
    parser.add_argument('dem_file', help='DEM数据文件')
    parser.add_argument('--step', type=int, default=5, help='DEM采样步长 (默认: 5)')
    parser.add_argument('--output-prefix', default='sim', help='输出文件前缀 (默认: sim)')
    parser.add_argument('--include-topographic-phase', action='store_true', help='包含地形相位')
    parser.add_argument('--baseline', type=float, default=0.0, help='垂直基线 (m)')
    parser.add_argument('--workers', type=int, default=None, help='并行进程数 (默认: CPU核心数)')
    parser.add_argument('--backscatter-model', default='cosine',
                        choices=['cosine', 'cosine2', 'constant', 'ohammers', 'brigham'],
                        help='后向散射模型')
    parser.add_argument('--noise', type=float, default=0.1, help='噪声水平 (默认: 0.1)')
    parser.add_argument('--layover-shadow', action='store_true', help='检测叠掩和阴影')
    parser.add_argument('--output-ext', default='.tif', help='输出文件扩展名')

    args = parser.parse_args()

    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)

    print("=== SAR模拟器 V3 ===")
    print(f"SAR参数: {args.yaml_file}")
    print(f"DEM文件: {args.dem_file}")

    sim = SARSimulatorV3(args.yaml_file, args.dem_file)
    sim.simulate(
        step=args.step,
        output_prefix=args.output_prefix,
        include_topographic_phase=args.include_topographic_phase,
        baseline_perpendicular=args.baseline,
        n_workers=args.workers,
        backscatter_model=args.backscatter_model,
        noise_level=args.noise,
        layover_shadow_detection=args.layover_shadow,
        output_ext=args.output_ext
    )

    print("\n=== 模拟完成 ===")


if __name__ == '__main__':
    main()
