# BeyondMimic Sim2Sim — Unitree H1 部署验证工具

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-green.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

一个轻量级的 **Sim-to-Sim 仿真验证工具**，用于在 MuJoCo 物理仿真器中部署和验证基于 [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) 框架训练的 Unitree H1 人形机器人全身动作追踪策略。

> 本项目是对 BeyondMimic 训练管道的 **部署端补充**，核心贡献是编写了一套完整的 MuJoCo 仿真回放脚本，
> 实现了从 ONNX 策略模型到物理仿真的完整闭环验证，无需安装 Isaac Lab 或 PyTorch。

## ✨ 功能特点

- 🤖 **多机器人支持**：统一架构支持 H1、G1、Go1、Go2 等多种机器人（`h1sim2sim.py` 专用于 H1，`sim2sim.py` 支持 G1/Go2）
- 📦 **纯 CPU 部署**：仅依赖 `mujoco` + `onnxruntime` + `numpy`，无需 GPU
- 🔄 **动态关节映射**：自动从 ONNX 元数据提取关节顺序，解决训练-部署关节顺序不一致问题
- 📐 **精确观测对齐**：110 维观测向量与训练环境 `TrackingEnvCfg` 完全一致
- 🎯 **真实锚点计算**：通过 `subtract_frame_transforms` 计算运动参考点的相对位姿
- 📝 **完整中文注释**：每个函数、每段逻辑都有详细的中文说明

## 📁 项目结构

```
beyondmimic-sim2sim-g1/
├── h1sim2sim.py             # H1 专用仿真脚本（带完整中文注释）
├── sim2sim.py               # G1/Go2 通用仿真脚本
├── README.md                # 本文件
├── LICENSE                  # MIT 许可证
├── .gitignore
├── requirements.txt         # Python 依赖
├── models/                  # ONNX 策略模型
│   ├── 2026-06-09_13-07-05_h1_walk.onnx  # H1 预训练的行走策略
│   └── dance_walk_01.onnx                # G1 预训练的舞步行走策略
├── motions/                 # 运动参考数据
│   ├── merged_walk1.npz     # H1 行走动作（NPZ 格式，含关节轨迹和刚体位姿）
│   ├── merged_walk1.csv     # H1 行走动作（CSV 格式原始数据）
│   ├── motion.npz           # G1 参考动作
│   └── dance1_subject1.csv  # G1 原始动作数据
├── assets/                  # MuJoCo 机器人模型
│   └── unitree_g1_mjcf/     # Unitree 机器人 MJCF 模型文件
│       ├── h1sum.xml        # H1 机器人模型
│       ├── g1.xml           # G1 机器人模型
│       ├── go1.xml          # Go1 机器人模型
│       ├── go2.xml          # Go2 机器人模型
│       └── meshes/h1/       # H1 机器人 STL 网格文件（46 个部件）
└── tools/                   # 辅助工具
    └── read_onnx.py         # ONNX 元数据查看工具
```

## 🚀 快速开始

### 环境安装

**Python 版本要求**: 3.11+

```bash
# 克隆仓库
git clone git@github.com:lijinpeng273-netizen/Beyondmimic_Sim2sim_h1.git
cd Beyondmimic_Sim2sim_h1

# 安装依赖
pip install -r requirements.txt
```

### 运行 H1 仿真

```bash
# 使用预训练的 H1 行走策略
python h1sim2sim.py \
    --motion_file motions/merged_walk1.npz \
    --xml_path assets/unitree_g1_mjcf/h1sum.xml \
    --policy_path models/2026-06-09_13-07-05_h1_walk.onnx

# 循环播放模式
python h1sim2sim.py \
    --motion_file motions/merged_walk1.npz \
    --xml_path assets/unitree_g1_mjcf/h1sum.xml \
    --policy_path models/2026-06-09_13-07-05_h1_walk.onnx \
    --loop

# 自定义控制分频（decimation = 每 N 个物理步推理一次）
python h1sim2sim.py \
    --motion_file motions/merged_walk1.npz \
    --xml_path assets/unitree_g1_mjcf/h1sum.xml \
    --policy_path models/2026-06-09_13-07-05_h1_walk.onnx \
    --decimation 4
```

### 命令行参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--motion_file` | str | ✅ | 运动参考数据路径（.npz 格式） |
| `--xml_path` | str | ✅ | MuJoCo XML 模型文件路径 |
| `--policy_path` | str | ✅ | ONNX 策略模型路径 |
| `--decimation` | int | ❌ | 控制分频，每 N 个物理步推理一次（默认 4，即 50Hz） |
| `--loop` | flag | ❌ | 动作序列结束后循环播放 |
| `--save_json` | flag | ❌ | 将运动数据导出为 JSON（调试用） |

## 🔧 技术架构

### 控制管道

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  运动参考数据  │────▶│  观测向量构造  │────▶│  ONNX 策略   │────▶│  PD 控制器    │
│  (.npz)      │     │   (110 维)    │     │  网络推理     │     │  力矩计算     │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                           ▲                                         │
                           │              ┌─────────────┐           │
                           └──────────────│  MuJoCo 仿真  │◀──────────┘
                            状态反馈       │   (200 Hz)   │
                                          └─────────────┘
```

### 观测向量布局（H1, 110维）

| 索引范围 | 维度 | 名称 | 数据来源 |
|----------|------|------|----------|
| `[0:38]` | 38 | `command` | 运动指令 = concat(目标关节位置, 目标关节速度) |
| `[38:41]` | 3 | `motion_anchor_pos_b` | 运动锚点在机器人局部坐标系中的相对位置 |
| `[41:47]` | 6 | `motion_anchor_ori_b` | 运动锚点相对朝向（6D旋转矩阵表示） |
| `[47:50]` | 3 | `base_lin_vel` | 基座线速度（机器人局部坐标系） |
| `[50:53]` | 3 | `base_ang_vel` | 基座角速度 |
| `[53:72]` | 19 | `joint_pos` | 关节位置偏差（当前 - 默认） |
| `[72:91]` | 19 | `joint_vel` | 关节速度 |
| `[91:110]` | 19 | `actions` | 上一步策略输出 |

### 关键设计决策

1. **动态关节映射**：ONNX 策略训练时的关节顺序与 MuJoCo XML 中的顺序不同，脚本在运行时自动进行双向重映射，避免力矩错配导致的物理爆炸。

2. **6D 旋转表示**：采用旋转矩阵的前两列（6维）而非四元数（4维）来表示朝向，因为 6D 表示在神经网络中具有更好的连续性。

3. **控制分频 (decimation=4)**：物理仿真频率 200Hz（dt=0.005s），策略推理频率 50Hz（每 4 个物理步推理一次），与训练配置严格一致。

4. **纯 NumPy 数学库**：用 NumPy 重新实现了 Isaac Lab 中的四元数运算（乘法、求逆、旋转）和坐标帧变换，摆脱了对 PyTorch 的依赖。

5. **IMU 传感器优先**：优先从 MuJoCo IMU 传感器读取基座姿态和角速度，回退到 free joint 数据，确保状态估计的准确性。

## 📊 H1 vs G1 参数对比

| 参数 | H1 | G1 |
|------|----|----|
| 观测向量维度 | 110 | 160 |
| 受控关节数 | 19 | 29 |
| 指令维度 | 38 | 58 |
| 参考刚体 | pelvis | pelvis |
| 初始骨盆高度 | 1.05 m | 0.80 m |
| 专用脚本 | `h1sim2sim.py` | `sim2sim.py` |
| MuJoCo 模型 | `h1sum.xml` | `g1.xml` |

## 📋 依赖项

| 包 | 版本 | 用途 |
|----|------|------|
| `mujoco` | ≥ 3.0 | 物理仿真引擎 |
| `onnx` | ≥ 1.14 | ONNX 模型加载与元数据解析 |
| `onnxruntime` | ≥ 1.16 | ONNX 模型推理（CPU） |
| `numpy` | ≥ 1.24 | 数值计算 |

## 🏗️ 使用自己训练的策略

1. 在 [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) 框架中训练好策略
2. 导出 ONNX 模型（训练脚本会自动嵌入关节名称、PD增益等元数据）
3. 将导出的 `.onnx` 文件和对应的 `.npz` 运动数据放入本项目
4. 运行 `h1sim2sim.py` 观察仿真效果

### 查看 ONNX 元数据

```bash
python tools/read_onnx.py
```

## 🔗 相关项目

- **[BeyondMimic (whole_body_tracking)](https://github.com/HybridRobotics/whole_body_tracking)** — 本项目所基于的动作追踪训练框架
- **[motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller)** — 官方的实机部署方案
- **[mjlab](https://github.com/mujocolab/mjlab)** — BeyondMimic 的 MuJoCo-Warp 替代实现
- **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)** — NVIDIA 的机器人学习框架

## 📜 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

本项目的 `h1sim2sim.py` 和 `sim2sim.py` 部署脚本为独立编写，参考了 BeyondMimic 的训练代码结构（MIT License）。
MuJoCo XML 模型文件来自 [Unitree Robotics](https://github.com/unitreerobotics) 的公开资源。

## 🙏 致谢

- [BeyondMimic](https://beyondmimic.github.io/) 团队提供的训练框架和算法
- [Unitree Robotics](https://www.unitree.com/) 提供的 H1、G1 机器人模型
- [MuJoCo](https://mujoco.org/) 物理仿真引擎
