# GMTSAR P2P 多视分辨率改造方案（自动/手动双分支）

## 1. 目标与约束

### 1.1 目标
- 提供两种分支：
  - 自动分辨率（默认）：根据 range/azimuth 地面分辨率自动计算 multilook。
  - 指定分辨率（手动）：支持显式输入 `rg:za`（含 `1:1`）。
- 对所有 p2p 相关流程统一行为，避免脚本间策略不一致。
- 最终 geocoding 分辨率必须满足：
  - 正方形像素。
  - 像素大小取“多视后 range/azimuth 地面分辨率中的较大者”。

### 1.2 当前主要问题（基于现代码）
- `p2p_processing.csh` 读取 `range_dec/azimuth_dec`，但未统一校验“是否同时存在”。
- `filter.csh` 对 DJ1（`SC==14`）存在硬覆盖：
  - 会把 `idec/jdec` 强制改为 `2/2`，覆盖外部传入多视。
- `geocode.csh`/`proj_ra2ll.csh` 的输出分辨率当前基于滤波波长（或 DJ1 特例），与“多视后地面分辨率最大值”目标不一致。
- `gmtsar/csh`、`bin`、`python/utils` 三套实现存在潜在漂移风险。

## 2. 总体设计

### 2.1 统一策略
- 引入统一“有效多视参数”概念：
  - `range_dec_eff`
  - `azimuth_dec_eff`
- 所有调用 `filter.csh` 的 p2p 流程仅使用 `*_eff`，不直接散用原始配置值。
- 所有 geocode 流程基于 `*_eff` 计算 geocoding 像素大小。

### 2.2 两种分支与优先级
- 分支 1：自动（默认）
  - 未显式指定 `rg:za` 时启用。
  - 自动计算 `range_dec_eff/azimuth_dec_eff`。
- 分支 2：手动
  - 显式指定 `rg:za` 时启用，`1:1` 合法。
  - 强制采用用户值，不再被任何卫星特例覆盖。

推荐优先级（高到低）：
1. CLI `rg:za`
2. 配置文件手动项（若启用手动模式）
3. 自动模式计算结果

## 3. 参数接口方案

### 3.1 CLI 接口（核心入口）
- `p2p_processing.csh SAT master aligned [config] [rg:za]`
- 支持：
  - 3参：默认自动
  - 4参：可能是 `config` 或 `rg:za`
  - 5参：`config + rg:za`

### 3.2 配置接口（建议新增）
- `multilook_mode = auto` 或 `manual`（默认 `auto`）
- `multilook_rg_az =`（格式 `rg:za`，仅 `manual` 使用）

兼容旧字段：
- `dec_factor` 继续保留。
- `range_dec/azimuth_dec` 可继续读取，但作为兼容输入，不再是唯一控制入口。

### 3.3 参数校验
- `rg:za` 正则：`^[0-9]+:[0-9]+$`
- 两个值均需 `>=1`
- 若只出现单边值（如仅 `range_dec`），直接报错并退出。

## 4. 自动多视算法

## 4.1 基础原则
- 目标：让多视后地面分辨率尽量满足
  - `range_dec_eff * dr_ground ~= azimuth_dec_eff * da_ground`
- 同时不低于原始较粗分辨率（避免无意义超采样）。

### 4.2 地面分辨率估计
优先方案：
- 从 `topo/trans.dat`（`r a topo lon lat`）估算地面 spacing：
  - 计算中心邻域内 `Δr=1` 对应地面距离中位数 `dr_ground`
  - 计算中心邻域内 `Δa=1` 对应地面距离中位数 `da_ground`

回退方案（无 `trans.dat` 时）：
- 使用 PRM 参数近似估计（`rng_samp_rate`、`PRF`、轨道/几何参数）。
- 若近似仍失败，回退到 `1:1` 并输出明确警告。

### 4.3 整数多视求解
- `target_ground = max(dr_ground, da_ground)`
- 初值：
  - `rg0 = round(target_ground / dr_ground)`
  - `az0 = round(target_ground / da_ground)`
- 在邻域（如 `±2`）搜索整数对，最小化：
  - `|rg * dr_ground - az * da_ground|`
- 约束：
  - `rg>=1, az>=1`
  - 可配置上限（如 `<=64`）防止过大窗口。

## 5. Geocoding 分辨率规则

### 5.1 规则
- 使用多视后地面分辨率：
  - `dr_post = dr_ground * range_dec_eff`
  - `da_post = da_ground * azimuth_dec_eff`
- geocoding 像素大小：
  - `geo_pix_m = max(dr_post, da_post)`
- 对所有输出统一使用正方形像素 `geo_pix_m x geo_pix_m`。

### 5.2 实现建议
- 新增一个中间文件（例如 `intf/<pair>/multilook.meta`）记录：
  - `dr_ground`
  - `da_ground`
  - `range_dec_eff`
  - `azimuth_dec_eff`
  - `geo_pix_m`
- `geocode.csh` 调用 `proj_ra2ll.csh` 时传 `geo_pix_m`，不再依赖 DJ1 特例常数。

## 6. 代码改造清单（按模块）

### 6.1 核心 CSH 流程（必须）
- `gmtsar/csh/p2p_processing.csh`
  - 扩展参数解析（`[config] [rg:za]`）
  - 计算 `range_dec_eff/azimuth_dec_eff`
  - Stage-4、iono 分支统一使用 `*_eff`
  - 记录并打印最终生效参数
- `gmtsar/csh/filter.csh`
  - 移除 DJ1（`SC==14`）多视硬覆盖逻辑，所有卫星统一走同一套 `*_eff` 策略
  - 保持旧接口兼容（5/6/7参）
- `gmtsar/csh/geocode.csh`
  - 读取 `geo_pix_m`
  - 所有 `proj_ra2ll.csh` 调用统一传入该值
- `gmtsar/csh/proj_ra2ll.csh`
  - 扩展参数支持“直接像素米值”输入（建议 `pix_m=<value>`）
  - 保留旧 `filter_wavelength` 行为兼容

### 6.2 所有调用 `filter.csh` 的 CSH 流程（需要同步策略）
- `gmtsar/csh/MAI_processing.csh`
- `gmtsar/csh/intf_tops.csh`
- `gmtsar/csh/intf_batch_ALOS2_SCAN.csh`
- `gmtsar/csh/batch_processing.csh`

要求：
- 全部接入统一 `*_eff` 计算/读取逻辑，避免与 p2p 主流程行为分叉。

### 6.3 卫星入口与路由（参数透传）
- `gmtsar/csh/p2p_mode_strip.csh`
- `gmtsar/csh/p2p_mode_tops.csh`
- `gmtsar/csh/p2p_sat_entry.csh`

要求：
- 入参数量放宽，支持 `rg:za` 透传。

### 6.4 Python utils（需同策略）
- `gmtsar/python/utils/p2p_processing`
- `gmtsar/python/utils/filter`
- `gmtsar/python/utils/geocode`
- `gmtsar/python/utils/proj_ra2ll`

要求：
- 与 CSH 逻辑一致，避免 CSH/Python 跑出不同结果。

### 6.5 发布镜像脚本目录同步
- 同步 `bin/` 对应脚本：
  - `bin/p2p_processing.csh`
  - `bin/filter.csh`
  - `bin/geocode.csh`
  - `bin/proj_ra2ll.csh`
  - `bin/p2p_mode_strip.csh`
  - `bin/p2p_mode_tops.csh`
  - `bin/p2p_sat_entry.csh`
  - 以及所有涉及 `filter.csh` 调用的脚本

## 7. 卫星统一处理策略（含 DJ1）

### 7.1 多视统一策略
- DJ1 不再单列，不保留默认硬编码多视。
- 所有卫星（含 DJ1）统一采用：
  - 手动模式：严格使用 `rg:za`
  - 自动模式：统一算法自动求解 `range_dec_eff/azimuth_dec_eff`

### 7.2 geocode 统一策略
- 不再保留任何 DJ1 专属 geocode 分支。
- 所有卫星统一走 `geo_pix_m = max(dr_post, da_post)`。

## 8. 兼容性与迁移

### 8.1 兼容老配置
- 老配置仅有 `dec_factor` 时，默认进入自动模式。
- 老配置写了 `range_dec/azimuth_dec`：
  - 若 `multilook_mode=manual` 或 CLI 指定手动，则按手动执行。
  - 否则仅作为参考，不覆盖自动结果。

### 8.2 失败回退
- 自动估算失败时，回退 `1:1`，并输出明确警告与建议。

## 9. 验证矩阵（必须执行）

### 9.1 功能验证
- 手动模式：
  - `1:1`
  - `2:2`
  - 非对称如 `8:2`
- 自动模式：
  - 不同传感器（至少 DJ1、S1_TOPS、ALOS2_SCAN、ERS）
  - 含/不含 `topo_phase` 路径

### 9.2 一致性验证
- CSH 与 Python utils 结果一致。
- `gmtsar/csh` 与 `bin` 脚本行为一致。

### 9.3 结果验证
- 检查 `range_dec_eff/azimuth_dec_eff` 是否与预期一致。
- 检查 geocode 输出像素是否正方形。
- 检查 geocode 像素是否等于多视后 r/a 地面分辨率较大者。

## 10. 分阶段实施建议

### Phase A（核心闭环）
- 改 `p2p_processing.csh + filter.csh + geocode.csh + proj_ra2ll.csh`
- 先在 `gmtsar/csh` 完成闭环，验证 DJ1 与 S1_TOPS（同一策略执行，无特例）。

### Phase B（全链覆盖）
- 同步所有 `filter.csh` 调用脚本与入口透传脚本。
- 同步 Python utils。

### Phase C（发布与文档）
- 同步 `bin/`。
- 更新 `pop_config.csh` 注释与 `config_examples` 示例。
- 增加“自动/手动多视”使用说明。

---

## 11. 本方案关键落地点（便于后续代码评审）
- 单一真值：`range_dec_eff/azimuth_dec_eff`
- 默认自动，手动覆盖
- DJ1 与其他卫星完全同策略处理（无任何专属多视/地理编码分支）
- geocoding 一律 `square + max(dr_post, da_post)`
- CSH/Python/bin 三套实现保持同策略
