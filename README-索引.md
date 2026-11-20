# 目录索引（多智能体协作版）

> 这是 `D:\DeepSeek\GPU-Compiler-memory` 在 2026-11-17 的完整快照，**之后所有代码更新在本目录进行**。
> 本文件是给人和智能体看的"入口地图"，不参与实验。

## 0. 防混淆约定（2026-11-17 用户定，全体智能体必须遵守）

1. **各自的报告** → `docs/agent-<名字>/`（如 `docs/agent-workbuddy/`、`docs/agent-deepseek/`）；
2. **各自跑出的日志** → `daima/oracle_logs/<日期>-<名字>/`——谁的数据一眼可辨；
3. **共享文件"单一所有者 + 只追加"**：
   - `daima/L1Feas.cpp`：**冻结**（v1.8.5 基线），谁都别改；
   - `docs/证据台账.md`：只许**追加行**；要改已有行必须在 `CHANGELOG.md` 留一行；
4. **根目录 `CHANGELOG.md`**：每次对共用文件的改动一行——时间 / 谁 / 改了什么 / 为什么。这是共用文件夹里唯一防"谁把数字改了"的机制。

## 1. 三个根目录的关系（重要）

| 路径 | 角色 | 谁写 |
|---|---|---|
| `D:\CC_GPU_1` | **论文根**：进度记忆（`memory/2026-11-15.md`）、路线文档（`wenjian/论文研究路线详解-1114.md`）以它为准 | 用户 / 人工同步 |
| **`C:\Users\Administrator\Desktop\GPU-Compiler\Memory optimization`** | **代码工作台**（本目录）：所有代码、脚本、分析产出在这里更新 | 智能体 |
| `D:\DeepSeek\GPU-Compiler-memory` | 旧工作区：只读留档，不再更新 | — |

## 2. 目录结构

| 目录 | 内容 | 能不能改 |
|---|---|---|
| `daima/` | 决策器 `L1Feas.cpp`、kernel `.mlir`、oracle 载具 `.cu`、回归脚本、扫参日志 | **冻结期**：`L1Feas.cpp` 不许动（v1.8.5 基线） |
| `daima/oracle_logs/2026-11-15/` | 11-15 之后的新日志（1024/bicg/干净512/探针/3mm 三轮/守卫日志） | 只增不改 |
| `daima/cuda_oracle/` | 载具源码快照（含服务器版 `atax_tiled.cu` 与旧版 `.bak`） | 只增不改 |
| `daima/xfer_2026-11-15/` | 11-15 迁移包（44 份日志 + 15 份 golden + 8 个补全的 `.mlir`） | 只增不改 |
| `docs/` | 证据台账、必要性重分析、B 线机理判读、审计计划 | 可更新（改后记日期） |
| `docs/evidence/`、`docs/prereg/`、`docs/holdout/` | 审计阶段产出（清单/预注册/留出规范） | 可更新 |
| `tools/` | 离线分析脚本（模型复现、五策略、稳定性、驻留、对账） | 可扩展，**不改已有脚本的判定口径** |
| `memory/` | 11-14 及以前的进度记忆（11-15 那份在 `D:\CC_GPU_1\memory\`） | 只读 |
| `wenjian/`、`paper/` | 文档与参考文献 PDF | 只读 |

## 3. 数据卫生红线（引用任何数字前必读）

以下日志**禁止进论文正文表**，只能作"污染/伪影"证据引用：

- `sweep_512_S1.log`、`sweep_512_S4.log` —— 整轮比干净会话慢约 12%
- `atax_sweep.log` —— acc 作用域 bug 那一轮（用 `atax_sweep2.log`）
- `heat1d_2^20.log`（非 `_redo`）—— 含 4 行 2.2 µs launch 失败假数据
- `3mm_rerun*.log` 的所有 <半机配置 —— 同会话抖动 20–70%
- `stril_symm_512_v3.log` —— 实为 syrk（`stril_tiled` 二进制是 syrk 专用）

干净卡基准：`ceshi-shuchu2/sweep_512_clean.log`（512³）与 `sweep_run1/2`、`sweep_friday`（4096³，跨会话 0.23%）。

## 4. 当前阶段与入口

- **当前阶段**：四臂裁判准备（离线包已就绪：spec v2 / 预注册 / 16 kernel / 四臂脚本 / manifest；待服务器跑留出批）。
- **入口文档**：`docs/prereg/2026-11-20-四臂裁判预注册.md`（预测与判据）、
  `docs/holdout/留出套件-spec-v2.md`（设计）、`docs/证据台账.md`（每个数字的来源与状态）。
- **已确定的三个决定**：规范根 = `D:\CC_GPU_1`（文档）/ 本目录（代码）；主指标 = 并列也算命中的 top-1；
  **留出批前不动** `L1Feas.cpp`（四臂结果后再落 v1.9 + F4，用户 11-20 已授权改码）。

## 5. 待补文件（在服务器上，尚未入档）

`twomm_tiled.cu`、`3mm_tiled.cu`、`mvt_tiled.cu`、`stril_tiled.cu`、服务器版 `bicg_tiled.cu`。
这 5 份是主表 5 行数字的来源，缺它们无法交 artifact。
