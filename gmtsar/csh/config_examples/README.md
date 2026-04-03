# separate_focus Config Examples

## Files
- `config.ERS.focus_only.txt`: ERS 全流程到聚焦结束（L0->L1）后退出。
- `config.ERS.focus_then_align.txt`: 在已完成预处理/聚焦基础上，从 Stage-2 继续做配准。
- `config.ALOS.focus_only.txt`: ALOS 的聚焦分离示例（L0->L1 后退出）。
- `config.S1_TOPS.standard.txt`: S1_TOPS 标准流程示例（`separate_focus` 对 TOPS 不生效）。
- `config.RS2.level1_from_stage2.txt`: 从 L1/SLC 数据开始（跳过 Stage-1）并从 Stage-2 配准开始。
- `config.DJ1.level1_from_stage2.txt`: DJ1 从 L1/SLC 数据开始（跳过 Stage-1）并从 Stage-2 配准开始。

## Typical usage
```bash
# 1) focus-only
p2p_processing.csh ERS <master> <slave> gmtsar/csh/config_examples/config.ERS.focus_only.txt

# 2) continue alignment
p2p_processing.csh ERS <master> <slave> gmtsar/csh/config_examples/config.ERS.focus_then_align.txt

# 3) L1/SLC start (example: DJ1)
p2p_processing.csh DJ1 <master> <slave> gmtsar/csh/config_examples/config.DJ1.level1_from_stage2.txt
```

## Notes
- `separate_focus = 1` 只对 `ERS/ENVI/ALOS/CSK_RAW` 生效。
- 若需要“聚焦后续跑配准”，建议配置：
  - `separate_focus = 0`
  - `proc_stage = 2`
  - `skip_stage = 1`
- `data_level = 1` 表示从 L1/SLC 数据起步：
  - 自动跳过 Stage-1 `pre_proc`
  - Stage-2 直接读取 `raw/*.PRM`, `raw/*.SLC`, `raw/*.LED`
  - 常用组合：`proc_stage = 2`, `skip_stage = 1`, `separate_focus = 0`
- 对 `S1_TOPS`，Stage-2 已独立到 `p2p_stage2_tops.csh`：
  - 会检查 `raw/<scene>.PRM/.SLC/.LED`（L1 起步必需）
  - 若 `correct_iono = 1`，还需要 `raw/<scene>.tiff/.xml/.EOF` 和 `topo/dem.grd`
- 可先运行最小预检（不执行处理运算）：
  - `tools/stage2_only_minimal_check.sh SAT master aligned <config> [WORKDIR]`
