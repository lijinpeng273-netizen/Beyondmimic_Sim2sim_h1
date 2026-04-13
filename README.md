# BeyondMimic Sim2Sim — Unitree G1 部署验证工具

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-green.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

一个轻量级的 **Sim-to-Sim 仿真验证工具**，用于在 MuJoCo 物理仿真器中部署和验证基于 [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) 框架训练的 Unitree G1 人形机器人全身动作追踪策略。

> 本项目是对 BeyondMimic 训练管道的 **部署端补充**，核心贡献是编写了一套完整的 MuJoCo 仿真回放脚本，
> 实现了从 ONNX 策略模型到物理仿真的完整闭环验证，无需安装 Isaac Lab 或 PyTorch。

## ✨ 功能特点

- 🤖 **多机器人支持**：统一架构支持 G1、HI、PI Plus 等多种人形机器人
- 📦 **纯 CPU 部署**：仅依赖 `mujoco` + `onnxruntime` + `numpy`，无需 GPU
- 🔄 **动态关节映射**：自动从 ONNX 元数据提取关节顺序，解决训练-部署关节顺序不一致问题
- 📐 **精确观测对齐**：160 维观测向量与训练环境 `TrackingEnvCfg` 完全一致
- 🎯 **真实锚点计算**：通过 `subtract_frame_transforms` 计算运动参考点的相对位姿
- 📝 **完整中文注释**：每个函数、每段逻辑都有详细的中文说明

## 📁 项目结构

```
beyondmimic-sim2sim-g1/
├── sim2sim.py              # 核心仿真脚本（带完整中文注释）
├── README.md               # 本文件
├── LICENSE                 # MIT 许可证
├── .gitignore
├── requirements.txt        # Python 依赖
├── models/                 # ONNX 策略模型
│   └── dance_walk_01.onnx  # 预训练的舞步行走策略
├── motions/                # 运动参考数据
│   ├── motion.npz          # NPZ 格式的参考动作（包含关节轨迹和刚体位姿）
│   └── dance1_subject1.csv # CSV 格式的原始动作数据
├── assets/                 # MuJoCo 机器人模型
│   └── unitree_g1_mjcf/    # Unitree G1 的 MJCF 模型文件
│       └── g1.xml
└── tools/                  # 辅助工具
    └── read_onnx.py        # ONNX 元数据查看工具
```

## 🚀 快速开始

### 环境安装

**Python 版本要求**: 3.11+

```bash
# 克隆仓库
git clone https://github.com/<your-username>/beyondmimic-sim2sim-g1.git
cd beyondmimic-sim2sim-g1

# 安装依赖
pip install -r requirements.txt
```

### 运行仿真

```bash
# 使用预训练的舞步行走策略
python sim2sim.py \
    --robot g1 \
    --motion_file motions/motion.npz \
    --xml_path assets/unitree_g1_mjcf/g1.xml \
    --policy_path models/dance_walk_01.onnx

# 循环播放模式
python sim2sim.py \
    --robot g1 \
    --motion_file motions/motion.npz \
    --xml_path assets/unitree_g1_mjcf/g1.xml \
    --policy_path models/dance_walk_01.onnx \
    --loop
```

### 命令行参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--robot` | str | ✅ | 机器人类型：`g1` / `hi` / `pi_plus` |
| `--motion_file` | str | ✅ | 运动参考数据路径（.npz 格式） |
| `--xml_path` | str | ✅ | MuJoCo XML 模型文件路径 |
| `--policy_path` | str | ✅ | ONNX 策略模型路径 |
| `--loop` | flag | ❌ | 动作序列结束后循环播放 |
| `--save_json` | flag | ❌ | 将运动数据导出为 JSON（调试用） |

## 🔧 技术架构

### 控制管道

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  运动参考数据  │────▶│  观测向量构造  │────▶│  ONNX 策略   │────▶│  PD 控制器    │
│  (motion.npz) │     │   (160 维)    │     │  网络推理     │     │  力矩计算     │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                           ▲                                         │
                           │              ┌─────────────┐           │
                           └──────────────│  MuJoCo 仿真  │◀──────────┘
                            状态反馈       │   (200 Hz)   │
                                          └─────────────┘
```

### 观测向量布局（G1, 160维）

| 索引范围 | 维度 | 名称 | 数据来源 |
|----------|------|------|----------|
| `[0:58]` | 58 | `command` | 运动指令 = concat(目标关节位置, 目标关节速度) |
| `[58:61]` | 3 | `motion_anchor_pos_b` | 运动锚点在机器人局部坐标系中的相对位置 |
| `[61:67]` | 6 | `motion_anchor_ori_b` | 运动锚点相对朝向（6D旋转矩阵表示） |
| `[67:70]` | 3 | `base_lin_vel` | 基座线速度（机器人局部坐标系） |
| `[70:73]` | 3 | `base_ang_vel` | 基座角速度 |
| `[73:102]` | 29 | `joint_pos` | 关节位置偏差（当前 - 默认） |
| `[102:131]` | 29 | `joint_vel` | 关节速度 |
| `[131:160]` | 29 | `actions` | 上一步策略输出 |

### 关键设计决策

1. **动态关节映射**：ONNX 策略训练时的关节顺序与 MuJoCo XML 中的顺序不同，脚本在运行时自动进行双向重映射，避免力矩错配导致的物理爆炸。

2. **6D 旋转表示**：采用旋转矩阵的前两列（6维）而非四元数（4维）来表示朝向，因为 6D 表示在神经网络中具有更好的连续性。

3. **控制分频 (decimation=4)**：物理仿真频率 200Hz（dt=0.005s），策略推理频率 50Hz（每 4 个物理步推理一次），与训练配置严格一致。

4. **纯 NumPy 数学库**：用 NumPy 重新实现了 Isaac Lab 中的四元数运算（乘法、求逆、旋转）和坐标帧变换，摆脱了对 PyTorch 的依赖。

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
4. 运行 `sim2sim.py` 观察仿真效果

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

本项目的 `sim2sim.py` 部署脚本为独立编写，参考了 BeyondMimic 的训练代码结构（MIT License）。
MuJoCo XML 模型文件来自 [Unitree Robotics](https://github.com/unitreerobotics) 的公开资源。

## 🙏 致谢

- [BeyondMimic](https://beyondmimic.github.io/) 团队提供的训练框架和算法
- [Unitree Robotics](https://www.unitree.com/) 提供的 G1 机器人模型
- [MuJoCo](https://mujoco.org/) 物理仿真引擎
