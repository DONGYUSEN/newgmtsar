# 卫星 Hook 开发说明（Stage-2/3/4）

## 1. 入口与命名

主流程会自动调用三层 Hook 分发器：

- `p2p_hook_stage2.csh`
- `p2p_hook_stage3.csh`
- `p2p_hook_stage4.csh`

若存在卫星专用脚本，则会优先分发到：

- `p2p_hook_stage2_<SAT>.csh`
- `p2p_hook_stage3_<SAT>.csh`
- `p2p_hook_stage4_<SAT>.csh`

例如：`p2p_hook_stage2_RS2.csh`。

## 2. 参数约定

- Stage-2：
  - `p2p_hook_stage2*.csh SAT when master aligned conf data_level skip_master`
- Stage-3：
  - `p2p_hook_stage3*.csh SAT when master aligned conf topo_phase shift_topo`
- Stage-4：
  - `p2p_hook_stage4*.csh SAT when ref rep conf topo_phase iono`

其中 `when` 取值：`pre` 或 `post`。

## 3. 返回码约定

- `pre` Hook：
  - `0`：继续默认阶段逻辑
  - `10`：跳过默认阶段逻辑
  - `20`：插件已完成该阶段逻辑（主流程不再执行默认逻辑）
  - 其他：错误并终止

- `post` Hook：
  - `0`：成功
  - 其他：错误并终止

## 4. 最小示例

```csh
#!/bin/csh -f
# p2p_hook_stage2_RS2.csh

set SAT = $1
set when = $2

if ($when == "pre") then
  # 例：仅打印信息，不改变流程
  echo "[RS2][Stage2][pre] custom hook"
  exit 0
endif

if ($when == "post") then
  echo "[RS2][Stage2][post] custom hook"
  exit 0
endif

exit 0
```
