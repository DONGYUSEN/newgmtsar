# GMTSAR DInSAR 配准与重采样模块

本目录用于主辅 SAR 图像配准与重采样，包含粗配准、精配准、偏移拟合、质量评估与重采样全流程。

## 目录结构

```text
registration/
├── coarse_regist.py          # FFT 粗配准
├── fine_regist.py            # 网格化精配准
├── offset_estimation.py      # 鲁棒多项式偏移拟合
├── registration_quality.py   # 配准质量评估
├── offset_proc.py            # 拟合 + 评估命令行封装
├── resampler.py              # sinc/kaiser 重采样
├── regist.py                 # 一站式主流程
└── README.md
```

## 推荐用法

先激活环境：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gmt
```

一站式运行（推荐）：

```bash
python3 regist.py master slave --output-dir ./output
```

说明：
- 输入需在当前目录下存在：`master.vrt`、`slave.vrt`、`master.yaml`、`slave.yaml`
- 中间文件统一写入：`./output/offset_estimate.txt`
- 输出文件：`slave_resamp.tiff`、`slave_resamp.vrt`、`slave_resamp.yaml`

## 分步运行

1. 粗配准
```bash
python3 coarse_regist.py master.vrt slave.vrt --window-size 256 --search-range 64
```

2. 精配准
```bash
python3 fine_regist.py master.vrt slave.vrt --grid-spacing 64 --num-workers 8 --output offset_estimate.txt
```

3. 偏移拟合与质量评估
```bash
python3 offset_proc.py --mode both --input offset_estimate.txt --output offset_estimate.txt --from-file
```

4. 重采样
```bash
python3 resampler.py master.yaml slave.yaml slave_resamp.yaml offset_estimate.txt --sinc-window-size 9 --num-workers 8
```

## 核心文件职责

- `coarse_regist.py`：2x2 窗口 FFT 粗配准，提供初始偏移并写入 `offset_estimate.txt`。
- `fine_regist.py`：全图网格局部相关，输出高密度配准点到 `offset_estimate.txt`。
- `offset_estimation.py`：对配准点做鲁棒拟合，生成 `PARAMETERS/NORMALIZATION` 多项式模型。
- `registration_quality.py`：基于偏移残差或图像对比评估质量等级与置信度。
- `offset_proc.py`：命令行封装，支持 `estimate`、`quality`、`both` 三种模式。
- `resampler.py`：读取偏移模型生成偏移场并重采样，输出 TIFF/VRT/YAML。
- `regist.py`：串联粗配准→精配准→拟合→评估→重采样。

## `offset_estimate.txt` 统一格式

当前流程统一使用同一个中间文件，包含以下段落：

1. 头行：`WINDOW_SIZE=... SEARCH_RANGE=... METHOD=...`
2. 点列表：`x dx y dy corr`
3. `FITTING_FORMULA`（拟合统计）
4. `PARAMETERS`（`a0..a5` 与 `b0..b5`）
5. `NORMALIZATION`（`rows_mean/std`、`cols_mean/std`）
6. `REGISTRATION_QUALITY`（`azimuth_rms`、`range_rms`、`num_points`）

`resampler.py` 同时兼容旧格式点列表；若缺少 `PARAMETERS`，会回退到常值偏移模型。

## 参数摘要

### `regist.py`
- 位置参数：`master_name`、`slave_name`
- 可选参数：`--output-dir`、`--window-size`、`--search-range`、`--correlation-threshold`、`--num-workers`

### `offset_proc.py`
- `--mode {estimate,quality,both}`
- `--input` 输入点文件（默认 `offset_estimate.txt`）
- `--output` 输出拟合结果文件（默认 `offset_estimate.txt`）
- `--quality-output` 质量评估输出（默认 `registration_quality.txt`）
- `--from-file` 从拟合结果文件评估质量
- `--reference` + `--registered` 直接图像评估

### `resampler.py`
- 位置参数：`master_yaml`、`slave_yaml`、`output_yaml`、`offset_file`
- 可选参数：`--sinc-window-size`、`--num-workers`、`--block-size`

## 注意事项

1. 推荐始终使用显式输出目录，避免多个场景共享同一个 `offset_estimate.txt`。
2. 若只运行分步脚本，请确保后一步读取的是同一份中间文件。
3. `slave.yaml` 中若 `metadata.data_file` 不是 `.vrt`，重采样脚本会自动尝试同名 `.vrt`。
4. 全流程中任一步失败都应先检查输入文件存在性与路径是否正确。

## 基于 `~/Temp/yaxia/output` 的最小可复现实测命令块

```bash
# 1) 激活环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gmt

# 2) 进入实测数据目录
cd ~/Temp/yaxia/output

# 3) 一站式全流程（推荐）
python3 ~/Software/GMTSAR/DInSAR/registration/regist.py master slave --output-dir int_reg_repro --num-workers 4

# 4) 关键结果快速检查
ls -lh int_reg_repro
head -n 5 int_reg_repro/offset_estimate.txt
grep -n "REGISTRATION_QUALITY\\|azimuth_rms\\|range_rms" int_reg_repro/offset_estimate.txt
```
