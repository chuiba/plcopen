# SO-ARM101 首个真机验证项目 — S5 批次计划

> 状态：**软件批次进行中（2026-07-26 维护者指定 SO-ARM101 为首个真机
> 验证项目）**。真机步骤以
> [feetech-sts-adapter-semantics.md](../compliance/feetech-sts-adapter-semantics.md)
> 的 **4.8 关卡**为闸（速度/加速度单位系数与 Status 位定义未锁定前，
> 一律不得声明真机完成——KB-074 边界原文"不进真机"）。本文件把
> [H1 载体计划](humanoid-h1-plan.md) S5/S6 行在 SO-ARM101 上具体化，
> 软件部分先行、硬件部分显式标注"硬件在手"前置。

## 1. 验证对象

| 项 | 事实 |
|---|---|
| 硬件 | SO-ARM101（TheRobotStudio / LeRobot 生态），6× Feetech **STS3215**（协议 0，型号号 777），TTL 总线 |
| 适配器现状 | `core/adapters/feetech.h`（KB-074，已批矩阵 S2 交付）：帧编解码 / FeetechBus 周期聚合 / FeetechServo / FeetechSim，全部无 IO |
| 生态定位 | LeRobot 遥操作 / 数据集 / VLA 策略 = stream 层（KB-035 H1 帧）的目标意图源；"VLA 出意图，plcopen 出运动"的第一块真机证据 |

## 2. 批次表

| 批 | 名 | 内容 | 前置 | DoD |
|----|----|------|------|-----|
| **SA1** ✅ | 4.8 核验工具面（软件，已交付） | `tools/feetech_verify.py`：总线扫描 / 寄存器读取 / **速度与加速度单位系数标定序列** / Status 位采样 / JSON 证据报告——即已批矩阵 §5"标定工具面（tools/，随 S5）"与 4.8 关卡的测量仪器。协议编解码与既有 C++ 黄金字节向量逐字节对齐，CI 以进程内寄存器映像假总线全流程测试 | 无 | 工具 CTest 全绿；黄金向量与 `feetech_tests.cpp` 一致；无硬件依赖 |
| **SA2** ✅ | 宿主串口 runner（软件 + dry-run，已交付：dry-run 1800/1800 响应、50Hz 占用 7.7%） | 宿主字节泵：`FeetechBus` tx/rx 缓冲 ↔ 串口（矩阵 2.1"传输留宿主"）；周期档 50Hz 起步（矩阵 2.8 预算）；`--dry-run` 走 FeetechSim 进 CI | SA1 | dry-run CTest 全绿；真机口径不声明 |
| **SA3** ✅ | 孪生对拍（软件，已交付：primitive 六关节拓扑 fixture + 静态契约/闭环双层测试） | T2b 通用装载器（1–48 hinge，本地 MJCF 路径）接 SO-ARM101 模型：维护者本地提供社区 MJCF（仓库不 vendor 厂商模型，KB-093 纪律），或仓库自有 primitive 六连杆 fixture 兜底 | SA2 | 同令流打 sim 的闭环 CTest；模型保真度不声明 |
| **SA4** ✅ | stream 升频演示（软件，已交付：ZOH vs 滤波 jerk 分离 268-303×、跟踪误差 ≤0.015 rad、棘轮门入 CTest） | 30–100Hz 意图流（数据集回放格式）→ `JointStreamGroup` upsample → runner/sim；量化 AB：裸下发 vs 经滤波的 jerk/平滑度/跟踪 | SA2/SA3 | AB 指标进测试；即 H1 计划 S6 的 LeRobot 互通评估素材 |
| **SA5** ✅ | LeRobot 数据集回放桥（软件，已交付：28 测试三层、对抗评审 8 项修复、分离 10.4-11.1×） | `tools/lerobot_replay.py`：episode 装载（LeRobot v2 parquet 目录/文件走**可选依赖** pyarrow/pandas；CSV/JSON 纯标准库兜底）→ 显式单位映射（`--scale/--offset/--position-limit`，4.8 未锁前工具不内置任何物理单位常数）→ 周期域帧（严格递增时间戳，KB-035 契约）→ `JointStreamSim` upsample 回放与 ZOH 对拍（SA4 同款三阶差分指标）。仓库自带合成 30fps 六关节 fixture；真实数据集由维护者本地提供路径，不 vendor（KB-093 同款纪律）。交付中定位了绑定路径分离比低于 SA4 C++ 面的根因——**KB-104**（StreamFilter1D 包络证明单调性假阴性，13.5× jerk 税；修复方向已验证、待语义批准） | SA4 | stdlib 层 CTest 全绿；sim 层入 twin CI；parquet 层无依赖时自跳过；真实数据集单位口径不声明 |
| **S5-HW** | 真机三步（**硬件在手**） | ① 用 SA1 工具在真机上执行 4.8 核验（单位系数 + Status 位锁定）→ ② KB-074 边界解除（声明变更：矩阵 2.6/2.11 待核项落数值，更新语义矩阵与 KB）→ ③ H1-S5 原义剧本（标定/MC_HomeDirect 回零/S3-S4 真机复现）与商用证据总账首行真机数据 | SA1-SA4 + 硬件 | 真机证据归档；此前一切"真机"表述禁用 |

## 3. 边界（本计划显式不声明）

- **4.8 关卡解除前不声明真机可用**——SA1-SA4 全部是软件与仿真交付。
- `pyserial` 仅为工具面**可选运行时依赖**（真机连接时才需要）；协议
  编解码纯标准库，CI 不引入新依赖。
- 不 vendor / 不下载厂商模型；MJCF 由维护者本地提供路径。
- 标定序列涉及真机运动：工具内置扭矩限幅与行程限制提示，但**机械
  安全始终由操作者负责**（与 STO/SS1 边界口径一致，软件不构成安全
  证明）。
- SO-ARM 级硬件不闭合商用门 #1/#4 的 PREEMPT_RT 指标——那属 B5/B6
  工业台架；本线的价值是真机证据链从零到一与 stream 定位实证。

## 4. 与既有登记的关系

- 已批矩阵 `feetech-sts-adapter-semantics.md`：2.6/2.11 待核项、4.8
  关卡、§5 工具面归属——本计划是其 S5 执行形态，不改动已批语义。
- KB-074（协议层边界）、KB-035（stream H1 帧）、KB-093（孪生装载纪律）
  ——各批交付时按 KB 纪律登记增量。
