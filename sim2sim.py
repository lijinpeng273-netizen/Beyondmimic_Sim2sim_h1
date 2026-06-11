from __future__ import annotations

"""统一的 Sim-to-Sim 仿真验证脚本，支持多种机器人配置。

本脚本的核心功能是：将强化学习训练得到的 ONNX 策略模型部署到 MuJoCo 物理仿真器中，
通过回放预录制的参考动作（.npz 文件），驱动策略网络实时推理并输出关节力矩，
从而在仿真环境中验证策略的动作追踪效果（即 "Sim-to-Sim" 验证）。

整体架构参考自 Mini-Pi-Plus 项目的统一化部署框架，
支持通过命令行参数动态切换机器人类型、动作文件、XML 模型和策略文件。

使用示例:
    # G1 机器人（Unitree G1 人形机器人）:
    python sim2sim.py --robot g1 --motion_file motions/motion.npz \
        --xml_path assets/unitree_g1_mjcf/g1.xml --policy_path models/dance_walk_01.onnx

    # 循环播放模式:
    python sim2sim.py --robot g1 --motion_file motions/motion.npz \
        --xml_path assets/unitree_g1_mjcf/g1.xml --policy_path models/dance_walk_01.onnx --loop
"""

import argparse
import json
import time
import os

import mujoco          # MuJoCo 物理引擎 Python 绑定
import mujoco.viewer   # MuJoCo 内置可视化窗口
import numpy as np     # 数值计算（替代 torch，实现纯 CPU 部署）
import onnx            # ONNX 模型加载与元数据解析
import onnxruntime     # ONNX 推理引擎（CPU 模式）

# ============================================================================
# 全局仿真参数
# ============================================================================
simulation_duration = 300.0   # 仿真总时长（秒），超过后自动退出
simulation_dt = 0.005         # 物理仿真步长（秒），与 Isaac Lab 训练时的 sim.dt 保持一致
control_decimation = 1        # 默认控制分频（每 N 个物理步执行一次策略推理），
                              # 具体值会被下方 ROBOT_CONFIGS 中的配置覆盖

# ============================================================================
# 机器人配置字典
# ============================================================================
# 每种机器人都有独立的配置项，包括关节数量、观测维度、参考刚体名称等。
# 当添加新机器人时，只需在此字典中新增一个条目即可。
ROBOT_CONFIGS = {
    "g1": {
        "num_actions": 29,      # G1 拥有 29 个受控关节（髋、膝、踝、腰、肩、肘、腕）
        "num_obs": 160,         # 观测向量总维度 = 58(指令) + 3(锚点位置) + 6(锚点朝向)
                                #   + 3(线速度) + 3(角速度) + 29(关节位置) + 29(关节速度) + 29(上一步动作)
        "reference_body": "pelvis",  # 参考刚体（锚点），用于计算运动指令的相对坐标
        "default_xml": None,         # 默认 XML 路径（None 表示需要命令行指定）
        "joint_names": None,         # 关节名称列表（None 表示从 ONNX 元数据动态加载）
        "motion_body_index": 0,      # 动作数据中锚点刚体的索引（后备值）
        "control_decimation": 4,     # 控制分频 = 4，与训练时 TrackingEnvCfg.decimation 完全一致
                                     # 即：每 4 个物理步（4 × 0.005s = 0.02s）执行一次策略推理
    },
    "hi": {
        "num_actions": 23,      # HI 机器人有 23 个受控关节
        "num_obs": 124,         # HI 的观测维度
        "reference_body": "base_link",
        "default_xml": None,
        "joint_names": [        # HI 的关节名称（硬编码，因为 ONNX 中可能不包含）
            "l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_thigh_joint", "l_hip_calf_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
            "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_thigh_joint", "r_hip_calf_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
            "waist_yaw_joint", "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint", "l_elbow_joint", "l_wrist_joint",
            "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint", "r_elbow_joint", "r_wrist_joint",
        ],
        "motion_body_index": 0,
        "control_decimation": 10,   # HI 使用 10 倍分频
    },
    "pi_plus": {
        "num_actions": 22,      # PI Plus 有 22 个受控关节
        "num_obs": 119,
        "reference_body": "base_link",
        "default_xml": None,
        "joint_names": [
            "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
            "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint", "l_elbow_joint", "l_wrist_joint",
            "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
            "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint", "r_elbow_joint", "r_wrist_joint",
        ],
        "motion_body_index": 0,
        "control_decimation": 10,
    }
}


# ============================================================================
# 数学工具函数
# ============================================================================
# 以下函数用纯 NumPy 实现了 Isaac Lab 中基于 PyTorch 的四元数/旋转矩阵运算，
# 使得本脚本无需安装 torch 和 scipy 即可在纯 CPU 环境下运行。

def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    """将四元数转换为旋转矩阵。
    
    四元数格式为 (w, x, y, z)，即标量在前。
    
    参数:
        quaternions: 形状为 (..., 4) 的四元数数组
    返回:
        形状为 (..., 3, 3) 的旋转矩阵
    """
    r, i, j, k = np.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),  # R[0,0]
            two_s * (i * j - k * r),       # R[0,1]
            two_s * (i * k + j * r),       # R[0,2]
            two_s * (i * j + k * r),       # R[1,0]
            1 - two_s * (i * i + k * k),   # R[1,1]
            two_s * (j * k - i * r),       # R[1,2]
            two_s * (i * k - j * r),       # R[2,0]
            two_s * (j * k + i * r),       # R[2,1]
            1 - two_s * (i * i + j * j),   # R[2,2]
        ),
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_obs(data):
    """从 MuJoCo 仿真数据中提取机器人的基础状态观测。
    
    提取内容包括：
    - qpos: 广义坐标（包含浮动基座位姿 + 关节角度）
    - dq:   广义速度
    - quat: 基座朝向四元数（优先从 IMU 传感器读取，不存在则从 qpos 回退）
    - v:    基座在 **自身坐标系** 下的线速度（通过旋转矩阵转换）
    - omega: 基座角速度
    - gvec: 重力在机器人坐标系下的投影（用于判断倾斜）
    - state_tau: 实际执行器力矩减去偏置力（科氏力、重力等）
    
    参数:
        data: mujoco.MjData 对象
    返回:
        (qpos, dq, quat, v, omega, gvec, state_tau) 元组
    """
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    
    # 尝试从 XML 中定义的 IMU 传感器读取朝向和角速度
    # 如果 XML 中没有定义传感器（如纯被动模型），则从 free joint 的 qpos/qvel 中回退读取
    try:
        quat = data.sensor("orientation").data[[0, 1, 2, 3]].astype(np.double)
        omega = data.sensor("angular-velocity").data.astype(np.double)
    except:
        quat = data.qpos[3:7].astype(np.double)   # free joint 中 qpos[3:7] 是四元数
        omega = data.qvel[3:6].astype(np.double)   # free joint 中 qvel[3:6] 是角速度
    
    # 将四元数转换为 3×3 旋转矩阵（使用 MuJoCo 原生 C 函数，性能最优）
    rotm = np.zeros(9)
    mujoco.mju_quat2Mat(rotm, quat)
    rotm = rotm.reshape((3, 3))
    
    # 将世界坐标系下的速度转换到机器人自身坐标系（R^T * v_world）
    v = (rotm.T @ data.qvel[:3]).astype(np.double)
    # 计算重力在机器人坐标系下的投影方向
    gvec = (rotm.T @ np.array([0.0, 0.0, -1.0])).astype(np.double)
    # 实际力矩 = 执行器输出 - 偏置力（重力 + 科氏力）
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)

    return (qpos, dq, quat, v, omega, gvec, state_tau)


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法（Hamilton 积）。
    
    使用了一种数值稳定的乘法算法，避免了直接基于分量展开时的精度损失。
    四元数格式：(w, x, y, z)。
    
    参数:
        q1, q2: 形状为 (..., 4) 的四元数数组，两者形状必须一致
    返回:
        q1 * q2 的结果，形状不变
    """
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """计算四元数的共轭。
    
    对于单位四元数，共轭等于逆。
    共轭运算：q* = (w, -x, -y, -z)
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)


def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """计算四元数的逆。
    
    q^{-1} = q* / |q|^2
    对于单位四元数（|q|=1），逆就等于共轭。
    eps 用于防止除零。
    """
    return quat_conjugate_np(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD 位置控制器：根据目标位置和当前位置计算关节力矩。
    
    公式：τ = Kp * (q_target - q_current) + Kd * (dq_target - dq_current)
    
    参数:
        target_q:  目标关节位置 (rad)
        q:         当前关节位置 (rad)
        kp:        比例增益（刚度, Nm/rad）
        target_dq: 目标关节速度（通常为 0）
        dq:        当前关节速度 (rad/s)
        kd:        微分增益（阻尼, Nm·s/rad）
    返回:
        各关节的控制力矩 (Nm)
    """
    return (target_q - q) * kp + (target_dq - dq) * kd


# ============================================================================
# 观测向量构造函数
# ============================================================================

def create_observation_hi_pi(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions):
    """构造 HI / PI Plus 机器人的观测向量。
    
    观测布局（以 HI 的 124 维为例）：
      [0:46]   motioninput        = 运动指令（目标关节位置 + 目标关节速度）
      [46:52]  motion_ref_ori_b   = 运动锚点相对朝向（旋转矩阵前 2 列展平，6 维）
      [52:55]  omega              = 基座角速度
      [55:78]  joint_pos_rel      = 关节位置 - 默认位置
      [78:101] joint_vel          = 关节速度
      [101:124] action_buffer     = 上一步的策略输出动作
    """
    cmd_size = len(motioninput)
    obs[offset:offset + cmd_size] = motioninput
    offset += cmd_size
    obs[offset:offset + 6] = motion_ref_ori_b
    offset += 6
    obs[offset:offset + 3] = omega
    offset += 3
    obs[offset:offset + num_actions] = qpos_seq - joint_pos_array_seq
    offset += num_actions
    obs[offset:offset + num_actions] = qvel_seq
    offset += num_actions   
    obs[offset:offset + num_actions] = action_buffer
    return obs


def quat_rotate_inverse_np(q, v):
    """用四元数的逆旋转来变换向量。
    
    等价于 Isaac Lab 中的 quat_rotate_inverse(q, v)，
    即将世界坐标系中的向量 v 转换到四元数 q 所描述的局部坐标系中。
    
    公式：v_local = R(q)^T * v = R(q^{-1}) * v
    """
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def subtract_frame_transforms_np(t01, q01, t02, q02):
    """计算坐标帧 2 相对于坐标帧 1 的变换，结果表示在坐标帧 1 的局部坐标系中。

    等价于 Isaac Lab 的 subtract_frame_transforms 函数：
      q_12 = quat_mul(quat_inv(q01), q02)    # 相对朝向
      t_12 = quat_rotate_inverse(q01, t02 - t01)  # 相对位置（在帧 1 局部坐标系中）

    物理含义：
      - t01, q01: 坐标帧 1（机器人锚点）的世界坐标位姿
      - t02, q02: 坐标帧 2（运动参考锚点）的世界坐标位姿
      - 返回值 t_12: 运动锚点在机器人锚点局部坐标系中的相对位置
      - 返回值 q_12: 运动锚点相对于机器人锚点的相对朝向

    参数:
        t01: 帧 1 的世界坐标位置 (3,)
        q01: 帧 1 的世界坐标四元数 (4,)  格式 (w,x,y,z)
        t02: 帧 2 的世界坐标位置 (3,)
        q02: 帧 2 的世界坐标四元数 (4,)
    返回:
        (t_12, q_12): 相对位置和相对四元数
    """
    q01_inv = quat_inv_np(q01)
    q_12 = quat_mul_np(q01_inv, q02)
    t_12 = quat_rotate_inverse_np(q01, t02 - t01)
    return t_12, q_12


def create_observation_g1(obs, motioninput, v, omega, qpos_seq, qvel_seq, action_buffer, 
                          default_joint_pos, robot_anchor_pos, robot_anchor_quat,
                          motion_anchor_pos, motion_anchor_quat):
    """构造 G1 机器人的观测向量，与训练环境 TrackingEnvCfg 完全对齐。
    
    观测布局（160 维，严格按照训练时的 ObservationsCfg.PolicyCfg 顺序排列）：
      [0:58]    command             = 运动指令 = concat(目标关节位置, 目标关节速度)
                                      来源: MotionCommand.command = cat(joint_pos, joint_vel)
      [58:61]   motion_anchor_pos_b = 运动锚点相对于机器人锚点的位置（机器人局部坐标系）
                                      来源: subtract_frame_transforms(robot, motion)[0]
      [61:67]   motion_anchor_ori_b = 运动锚点的相对朝向（6D 旋转表示 = 旋转矩阵前 2 列展平）
                                      来源: matrix_from_quat(q_12)[..., :2].reshape(-1)
      [67:70]   base_lin_vel        = 机器人基座线速度（局部坐标系）
      [70:73]   base_ang_vel        = 机器人基座角速度（局部坐标系）
      [73:102]  joint_pos           = 关节位置偏差（当前位置 - 默认位置）
      [102:131] joint_vel           = 关节速度
      [131:160] actions             = 上一步策略输出（动作缓冲区）

    参数:
        obs:                预分配的观测数组
        motioninput:        运动指令 = concat(joint_pos[t], joint_vel[t])
        v:                  基座线速度（机器人局部坐标系）
        omega:              基座角速度（机器人局部坐标系）
        qpos_seq:           当前关节位置（ONNX 策略期望的顺序）
        qvel_seq:           当前关节速度（ONNX 策略期望的顺序）
        action_buffer:      上一步策略输出
        default_joint_pos:  默认关节位置（来自 ONNX 元数据的 default_joint_pos）
        robot_anchor_pos:   机器人锚点刚体的世界坐标位置
        robot_anchor_quat:  机器人锚点刚体的世界坐标四元数
        motion_anchor_pos:  运动参考数据中锚点的世界坐标位置
        motion_anchor_quat: 运动参考数据中锚点的世界坐标四元数
    返回:
        填充后的观测数组
    """
    # 1. 运动指令：目标关节位置 + 目标关节速度（共 29×2 = 58 维）
    cmd = motioninput if len(motioninput) == len(qpos_seq) * 2 else np.zeros(len(qpos_seq) * 2)

    # 2. 计算运动锚点在机器人局部坐标系中的相对位姿
    #    调用 subtract_frame_transforms 得到相对位置 (3维) 和相对四元数 (4维)
    anchor_pos_b, anchor_quat_b = subtract_frame_transforms_np(
        robot_anchor_pos, robot_anchor_quat, motion_anchor_pos, motion_anchor_quat
    )

    # 3. 将相对四元数转换为 6D 旋转表示（旋转矩阵的前 2 列展平）
    #    这种表示方式比欧拉角或四元数更适合神经网络学习（连续性更好）
    mat = matrix_from_quat(anchor_quat_b)  # (3, 3) 旋转矩阵
    anchor_ori_b = mat[..., :2].reshape(-1)  # 取前 2 列并展平 → (6,)

    # 按照训练时的严格顺序拼接所有观测分量
    obs_list = [
        cmd,              # 58 维：运动指令
        anchor_pos_b,     # 3  维：运动锚点相对位置
        anchor_ori_b,     # 6  维：运动锚点相对朝向（6D 表示）
        v,                # 3  维：基座线速度（局部坐标系）
        omega,            # 3  维：基座角速度（局部坐标系）
        qpos_seq - default_joint_pos,  # 29 维：关节位置偏差
        qvel_seq,         # 29 维：关节速度
        action_buffer     # 29 维：上一步动作
    ]
    
    obs_array = np.concatenate(obs_list).astype(np.float32)
    
    # 动态调整观测数组大小（防止预分配的 obs 过小）
    if len(obs) < len(obs_array):
        obs = np.zeros(len(obs_array), dtype=np.float32)
        
    obs[:len(obs_array)] = obs_array
    return obs


# ============================================================================
# ONNX 元数据解析工具
# ============================================================================
# ONNX 模型的 metadata_props 中存储了训练时的关键配置信息，
# 包括关节名称、默认位置、PD 增益、动作缩放等参数。
# 这些解析器负责将字符串形式的元数据转换为 Python 列表/数组。

def parse_str_list(val):
    """解析字符串列表。支持 JSON 格式和逗号分隔格式。
    
    示例输入: "['joint_a', 'joint_b']" 或 "joint_a, joint_b"
    """
    if not val: return []
    try:
        parsed = json.loads(val.replace("'", '"'))
        if isinstance(parsed, list): return parsed
    except: pass
    return [x.strip() for x in val.split(',')]


def parse_csv_list(val, dtype=float):
    """解析数值列表。支持 JSON 格式和逗号分隔格式。
    
    示例输入: "[0.548, 0.351, 0.439]" 或 "0.548, 0.351, 0.439"
    """
    if not val: return []
    try:
        parsed = json.loads(val.replace("'", '"'))
        if isinstance(parsed, list):
            return [dtype(x) if isinstance(x, (int, float, str)) else x for x in parsed]
    except: pass
    return [dtype(x.strip()) for x in val.split(',')]


# ============================================================================
# 运动锚点索引解析
# ============================================================================

def resolve_motion_anchor_index(
    motion_body_count: int,
    model_body_count: int,
    anchor_body_id: int,
    anchor_body_name: str | None,
    metadata_body_names: list[str] | None,
    fallback_index: int,
) -> int:
    """解析运动数据中锚点刚体的索引。

    运动文件（.npz）中的 body_pos_w 数组的第二个维度可能对应：
      情况 1：ONNX 元数据中 body_names 指定的一组筛选后的刚体（顺序与导出时一致）
      情况 2：MuJoCo 模型中的所有刚体（不含 world body）
      情况 3：MuJoCo 模型中的所有刚体（含 world body）

    本函数依次尝试上述三种匹配方式，找到锚点刚体在运动数组中的正确索引。

    参数:
        motion_body_count:  运动数据中刚体数量（motionpos.shape[1]）
        model_body_count:   MuJoCo 模型中刚体总数（m.nbody，含 world body）
        anchor_body_id:     锚点刚体在 MuJoCo 模型中的 ID
        anchor_body_name:   锚点刚体名称
        metadata_body_names: ONNX 元数据中的 body_names 列表（可能为 None）
        fallback_index:     匹配失败时的后备索引
    返回:
        锚点刚体在运动数组中的索引
    """
    # 情况 1：运动数据存储的是 ONNX 元数据中指定的筛选后刚体集合
    if anchor_body_name and metadata_body_names and motion_body_count == len(metadata_body_names):
        if anchor_body_name in metadata_body_names:
            return metadata_body_names.index(anchor_body_name)

    # 情况 2：运动数据不含 world body（最常见的导出方式）
    if anchor_body_id >= 0 and motion_body_count == model_body_count - 1:
        return anchor_body_id - 1

    # 情况 3：运动数据包含 world body
    if anchor_body_id >= 0 and motion_body_count == model_body_count:
        return anchor_body_id

    # 回退到配置中的默认索引
    return fallback_index


# ============================================================================
# 主仿真循环
# ============================================================================

def run_simulation(robot_type: str, motion_file: str, xml_path: str, policy_path: str, save_json: bool = False, loop: bool = False):
    """运行 Sim-to-Sim 仿真验证。
    
    完整流程：
    1. 加载运动数据（.npz）→ 提取参考关节轨迹和刚体位姿
    2. 加载 ONNX 策略模型 → 提取元数据（关节名称、PD 增益、动作缩放等）
    3. 加载 MuJoCo XML 模型 → 创建物理仿真环境
    4. 建立关节名称映射：ONNX 顺序 ↔ XML 顺序（关键！两者顺序不同）
    5. 进入仿真主循环：
       a. 读取机器人状态
       b. 每隔 control_decimation 步执行一次策略推理
       c. 将策略输出经 PD 控制器转换为力矩
       d. 施加力矩并步进物理仿真
    
    参数:
        robot_type:  机器人类型 ("g1" / "hi" / "pi_plus")
        motion_file: 运动数据文件路径（.npz 格式），或任意路径触发 dummy 回退
        xml_path:    MuJoCo XML 模型路径
        policy_path: ONNX 策略文件路径
        save_json:   是否将运动数据另存为 JSON（用于调试/可视化）
        loop:        是否在运动序列结束后循环播放
    """
    config = ROBOT_CONFIGS[robot_type]
    print(f"[INFO]: Using robot configuration: {robot_type}")
    
    # ========================================================================
    # 第一步：加载运动数据
    # ========================================================================
    # npz 文件包含以下数组：
    #   - body_pos_w:  (T, B, 3) 各刚体的世界坐标位置
    #   - body_quat_w: (T, B, 4) 各刚体的世界坐标四元数
    #   - joint_pos:   (T, J)    各关节的目标角度（作为策略输入的"指令"）
    #   - joint_vel:   (T, J)    各关节的目标角速度
    # 其中 T=帧数, B=刚体数, J=关节数
    try:
        motion = np.load(motion_file)
        motionpos = motion["body_pos_w"]
        motionquat = motion["body_quat_w"]
        motioninputpos = motion["joint_pos"]
        motioninputvel = motion["joint_vel"]
        num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0], motionpos.shape[0], motionquat.shape[0])
    except:
        # 文件无法读取时，使用全零的 dummy 数据（机器人将保持初始姿态）
        print("[WARNING]: Motion file unreadable or invalid, falling back to dummy sequence.")
        num_frames = 1000
        dummy_joints = config.get("num_actions", 29)
        motionpos = np.zeros((num_frames, 1, 3))
        motionquat = np.zeros((num_frames, 1, 4))
        motioninputpos = np.zeros((num_frames, dummy_joints))
        motioninputvel = np.zeros((num_frames, dummy_joints))
    
    def frame_idx(t):
        """根据当前时间步获取运动帧索引（支持循环模式）。"""
        if loop and num_frames > 0:
            return t % num_frames              # 循环模式：取模
        return t if t < num_frames else num_frames - 1  # 非循环：钳位到最后一帧
    
    # 可选：将运动数据导出为 JSON（用于外部工具可视化）
    if save_json:
        motion_dict = {
            "body_pos_w": motionpos.tolist(),
            "body_quat_w": motionquat.tolist(),
            "joint_pos": motioninputpos.tolist(),
            "joint_vel": motioninputvel.tolist()
        }
        motion_dir = os.path.dirname(motion_file)
        motion_basename = os.path.basename(motion_file)
        if motion_dir.endswith('/npz') or motion_dir.endswith('\\npz'):
            json_dir = motion_dir[:-3] + 'json'
        else:
            json_dir = motion_dir
        os.makedirs(json_dir, exist_ok=True)
        json_filename = os.path.join(json_dir, motion_basename.replace('.npz', '.json'))
        with open(json_filename, 'w') as f:
            json.dump(motion_dict, f, indent=2)
        print(f"[INFO]: Motion data saved to: {json_filename}")
    
    # ========================================================================
    # 第二步：加载 ONNX 策略模型并提取元数据
    # ========================================================================
    # ONNX 模型的 metadata_props 中嵌入了训练时的完整配置信息，
    # 使得部署端无需额外的 YAML 配置文件即可重建控制管道。
    model = onnx.load(policy_path)
    joint_seq = None              # ONNX 期望的关节顺序（策略训练时的顺序）
    joint_pos_array_seq = None    # 默认关节位置（ONNX 顺序）
    stiffness_array_seq = None    # 关节刚度 Kp（ONNX 顺序）
    damping_array_seq = None      # 关节阻尼 Kd（ONNX 顺序）
    action_scale = None           # 动作缩放因子（将策略输出映射到实际弧度偏移）
    anchor_body_name = None       # 锚点刚体名称（可能嵌入在 ONNX 中）
    metadata_body_names = None    # ONNX 中记录的刚体名称列表
    
    for prop in model.metadata_props:
        if prop.key == "joint_names":
            # 关节名称列表，定义了策略输入/输出的关节顺序
            joint_seq = parse_str_list(prop.value)
        elif prop.key == "default_joint_pos":
            # 默认关节角度（弧度），策略输出是相对于此的偏移量
            joint_pos_array_seq = np.array(parse_csv_list(prop.value))
        elif prop.key == "joint_stiffness":
            # PD 控制器的比例增益 Kp（Nm/rad）
            stiffness_array_seq = np.array(parse_csv_list(prop.value))
        elif prop.key == "joint_damping":
            # PD 控制器的微分增益 Kd（Nm·s/rad）
            damping_array_seq = np.array(parse_csv_list(prop.value))
        elif prop.key == "action_scale":
            # 动作缩放系数：target_pos = action * scale + default_pos
            action_scale = np.array(parse_csv_list(prop.value))
        elif prop.key == "anchor_body_name":
            # 锚点刚体名称（如 "pelvis"）
            anchor_body_name = prop.value
        elif prop.key == "body_names":
            # 训练时使用的刚体名称列表（用于解析运动数据中的刚体索引）
            metadata_body_names = parse_str_list(prop.value)
        elif prop.key == "observation_names":
            # 观测分量名称列表（仅用于文档说明，不影响逻辑）
            pass

    # ========================================================================
    # 第三步：加载 MuJoCo 物理模型
    # ========================================================================
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt  # 设置物理仿真步长

    # ========================================================================
    # 第四步：建立关节名称映射（ONNX 顺序 ↔ XML 顺序）
    # ========================================================================
    # 【关键概念】ONNX 策略和 MuJoCo XML 中的关节顺序通常不同！
    #
    # 例如：
    #   ONNX 顺序: [left_hip_pitch, right_hip_pitch, waist_yaw, ...]
    #   XML 顺序:  [left_hip_pitch, left_hip_roll, left_hip_yaw, ...]
    #
    # 如果不做映射，PD 控制器会把"右腿"的力矩施加到"左膝"上，导致物理爆炸！
    #
    # 术语约定：
    #   joint_seq:  ONNX 策略训练时的关节顺序（策略输入/输出使用此顺序）
    #   joint_xml:  MuJoCo XML 模型中的关节顺序（qpos/qvel 使用此顺序）
    
    if joint_seq is None:
        joint_seq = config["joint_names"]
    if config["joint_names"] is None:
        # 从 XML 模型中动态提取关节名称（跳过索引 0 的 free joint）
        config["joint_names"] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
        config["num_actions"] = len(joint_seq)
        
    print(f"[INFO]: Actions: {config['num_actions']}, Observations expected by ONNX Metadata: {config['num_obs']}")

    # 将 ONNX 顺序的参数重映射到 XML 顺序
    # 例：joint_pos_array[i] = joint_pos_array_seq[joint_seq.index(joint_xml[i])]
    joint_xml = config["joint_names"]
    joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    
    # PD 增益重映射（如果 ONNX 中未嵌入，则使用零值，策略输出将直接作为位置目标）
    if stiffness_array_seq is None or len(stiffness_array_seq) == 0:
        print("[WARNING] Missing joint_stiffness in ONNX metadata. Using PD targets directly.")
        stiffness_array = np.zeros(len(joint_xml))
        damping_array = np.zeros(len(joint_xml))
    else:
        stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
        damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])
    if action_scale is None or len(action_scale) == 0:
        action_scale = np.ones(len(joint_xml))
        
    print("action_scale", action_scale)

    # ========================================================================
    # 第五步：解析锚点刚体索引
    # ========================================================================
    # 锚点刚体（anchor body）是整个动作追踪系统的"坐标原点"，
    # 通常选择骨盆（pelvis）或基座（base_link）。
    # 策略通过观测 "运动锚点相对于机器人锚点的偏差" 来决定如何运动。
    reference_body = anchor_body_name or config["reference_body"]
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, reference_body)
    if body_id == -1:
        raise ValueError(f"Reference body '{reference_body}' not found in model {xml_path}.")

    # 解析运动数据中锚点刚体的索引（可能与 MuJoCo 模型中的索引不同）
    motion_body_idx = resolve_motion_anchor_index(
        motion_body_count=motionpos.shape[1],
        model_body_count=m.nbody,
        anchor_body_id=body_id,
        anchor_body_name=reference_body,
        metadata_body_names=metadata_body_names,
        fallback_index=config["motion_body_index"],
    )
    print(f"[INFO]: Reference body: {reference_body}, robot body id: {body_id}, motion body index: {motion_body_idx}")
    
    # ========================================================================
    # 第六步：初始化仿真状态
    # ========================================================================
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    control_decimation = config.get("control_decimation", 10)
    
    # 加载 ONNX 推理会话（使用 CPU 推理引擎）
    policy = onnxruntime.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
    input_shape = policy.get_inputs()[0].shape  # 期望的输入形状，如 [1, 160]
    
    action_buffer = np.zeros((num_actions,), dtype=np.float32)  # 上一步动作缓冲
    timestep = 0  # 运动序列时间步索引
    
    # 安全截断运动输入（处理关节数量不匹配的情况）
    minp = motioninputpos[frame_idx(timestep), :]
    minv = motioninputvel[frame_idx(timestep), :]
    if len(minp) > num_actions: minp = minp[:num_actions]
    if len(minv) > num_actions: minv = minv[:num_actions]
    motioninput = np.concatenate((minp, minv), axis=0)
    
    motionposcurrent = motionpos[frame_idx(timestep), motion_body_idx, :]
    motionquatcurrent = motionquat[frame_idx(timestep), motion_body_idx, :]
    
    # 设置初始关节位置为默认位置（让机器人从标准站立姿态开始）
    target_dof_pos = joint_pos_array.copy()
    if robot_type == "g1":
        d.qpos[2] = 0.76       # G1 站立时骨盆离地高度（米）
    elif robot_type == "hi":
        d.qpos[2] = 0.68       # HI 站立时基座高度
        
    if len(d.qpos) > 7:
        d.qpos[7:7+len(joint_xml)] = joint_pos_array  # 设置各关节到默认角度
    d.qvel[:] = 0.0             # 初始速度为零
    if m.nu > 0:
        d.ctrl[:] = 0.0         # 清零执行器控制信号
    d.qfrc_applied[:] = 0.0     # 清零外部施加力
    mujoco.mj_forward(m, d)     # 执行正向运动学：根据 qpos 计算 xpos, xquat 等

    # ========================================================================
    # 第七步：主仿真循环
    # ========================================================================
    # 循环结构：
    #
    #   while 仿真运行中:
    #       1. 读取当前机器人状态（get_obs）
    #       2. 每 control_decimation 步执行一次策略推理
    #          a. 构造观测向量（160 维）
    #          b. 送入 ONNX 策略网络推理
    #          c. 将策略输出转换为目标关节位置
    #       3. 通过 PD 控制器计算关节力矩
    #       4. 施加力矩并步进物理仿真
    #       5. 同步可视化窗口
    #
    # 关于控制分频（decimation）：
    #   训练时 dt=0.005s, decimation=4，即策略推理频率 = 1/(0.005×4) = 50Hz
    #   物理仿真频率 = 1/0.005 = 200Hz（PD 控制在 200Hz 下持续输出力矩）
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # 读取当前机器人状态
            qpos, dq, quat, v, omega, gvec, state_tau = get_obs(d)

            # ------ 策略推理（每 control_decimation 步执行一次）------
            if counter % control_decimation == 0:
                idx = frame_idx(timestep)
                
                # 从运动数据中获取当前帧的关节指令
                minp = motioninputpos[idx, :]
                minv = motioninputvel[idx, :]
                if len(minp) > num_actions: minp = minp[:num_actions]
                if len(minv) > num_actions: minv = minv[:num_actions]
                motioninput = np.concatenate((minp, minv), axis=0)  # 运动指令 = [pos, vel]
                
                # 获取当前帧运动数据中锚点的朝向（用于计算相对位姿）
                motionquatcurrent = motionquat[idx, motion_body_idx, :]
                
                # 从 MuJoCo 的 qpos/qvel 中提取关节状态，并重映射到 ONNX 期望的顺序
                # qpos[0:7] = free joint (位置 + 四元数), qpos[7:] = 关节角度
                # qvel[0:6] = free joint (线速度 + 角速度), qvel[6:] = 关节角速度
                qpos_xml = d.qpos[7:7 + num_actions]
                qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                qvel_xml = d.qvel[6:6 + num_actions]
                qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                
                # 根据机器人类型构造观测向量
                if robot_type == "g1":
                    # 获取机器人锚点（pelvis）的世界坐标位姿
                    # d.xpos[body_id] 是 mj_forward/mj_step 计算出的刚体世界坐标位置
                    robot_anchor_pos = d.xpos[body_id].copy()
                    robot_anchor_quat = d.xquat[body_id].copy()  # MuJoCo 四元数格式: (w,x,y,z)
                    # 获取运动数据中对应锚点的位姿
                    motion_anchor_pos = motionpos[idx, motion_body_idx, :].copy()
                    motion_anchor_quat = motionquat[idx, motion_body_idx, :].copy()
                    
                    # 构造 160 维观测向量
                    obs = create_observation_g1(obs, motioninput, v, omega, qpos_seq, qvel_seq, action_buffer, 
                                                joint_pos_array_seq, robot_anchor_pos, robot_anchor_quat,
                                                motion_anchor_pos, motion_anchor_quat)
                elif robot_type in ["hi", "pi_plus"]:
                    # HI/PI Plus 使用不同的观测构造方式（四元数相对旋转 → 6D 表示）
                    q01 = quat                     # 机器人朝向
                    q02 = motionquatcurrent         # 运动参考朝向
                    q10 = quat_inv_np(q01)          # 机器人朝向的逆
                    if q02 is not None:
                        q12 = quat_mul_np(q10, q02) # 相对旋转四元数
                    else:
                        q12 = q10
                    mat = matrix_from_quat(q12)
                    motion_ref_ori_b = mat[..., :2].reshape(6)  # 6D 旋转表示
                    
                    offset = 0
                    obs = create_observation_hi_pi(obs, offset, motioninput, motion_ref_ori_b, omega, qpos_seq, qvel_seq, action_buffer, joint_pos_array_seq, num_actions)
                
                # 处理观测维度与 ONNX 模型期望维度不完全匹配的情况
                target_dim = input_shape[1]
                if obs.shape[0] < target_dim:
                    obs = np.pad(obs, (0, target_dim - obs.shape[0]))  # 不足则零填充
                elif obs.shape[0] > target_dim:
                    obs = obs[:target_dim]  # 多余则截断

                # ------ ONNX 策略网络推理 ------
                # 策略网络接收两个输入：
                #   1. obs:       观测向量 (1, 160)
                #   2. time_step: 当前运动帧索引 (1, 1)，策略需要知道动作进行到哪了
                obs_tensor = np.expand_dims(obs, axis=0)
                action_out = policy.run(['actions'], {
                    policy.get_inputs()[0].name: obs_tensor,
                    policy.get_inputs()[1].name: np.array([[frame_idx(timestep)]], dtype=np.float32)
                })[0]
                
                # 将策略输出转换为目标关节位置
                # 公式：target_pos = action * action_scale + default_pos
                # action_scale 控制了策略输出的弧度范围（通常 0.1~0.5 rad）
                action_array = np.asarray(action_out).reshape(-1)
                action_buffer = action_array.copy()  # 存入缓冲，下一步作为观测输入
                target_dof_seq = action_array * action_scale + joint_pos_array_seq  # ONNX 顺序
                target_dof_seq = target_dof_seq.reshape(-1,)
                # 重映射目标位置：从 ONNX 顺序 → XML 顺序（用于 PD 控制器）
                target_dof_pos = np.array([target_dof_seq[joint_seq.index(joint)] for joint in joint_xml])
                
                # 推进运动序列时间步
                if loop or timestep + 1 < num_frames:
                    timestep += 1

            # ------ PD 控制 + 物理步进 ------
            # 无论是否执行了策略推理，每个物理步都要施加 PD 控制力矩
            if np.any(stiffness_array > 0):
                # 使用 PD 控制器计算关节力矩
                tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)
                if m.nu == num_actions:
                    # XML 中定义了执行器 (actuator)：通过 d.ctrl 施加
                    d.ctrl[:] = tau
                else:
                    # XML 中没有执行器（纯被动模型）：直接在广义力中施加
                    # 先清零施加力（避免上一步残留），再写入关节力矩
                    d.qfrc_applied[:] = 0.0
                    d.qfrc_applied[6:6+num_actions] = tau  # 跳过前 6 个自由度（free joint 的力/力矩）
            else:
                # 无 PD 增益：直接将目标位置作为 ctrl（适用于有位置伺服的 XML）
                if m.nu == num_actions:
                    d.ctrl[:] = target_dof_pos
                else:
                    d.qfrc_applied[:] = 0.0  # 无法施加力矩，保持自然运动

            # 步进物理仿真（1 步 = simulation_dt 秒）
            mujoco.mj_step(m, d)
            counter += 1

            # 同步可视化窗口（将仿真状态渲染到 MuJoCo Viewer 中）
            viewer.sync()

            # 实时时间同步：确保仿真速度不超过真实时间（接近 1:1 实时）
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    """解析命令行参数并启动仿真。"""
    parser = argparse.ArgumentParser(description="统一的 Sim-to-Sim 仿真验证脚本，支持多种机器人配置。")
    parser.add_argument("--robot", type=str, choices=["hi", "pi_plus", "g1"], required=True,
                        help="机器人类型: hi (高力矩), pi_plus (PI Plus), g1 (Unitree G1)")
    parser.add_argument("--motion_file", type=str, required=True, 
                        help="运动数据 NPZ 文件路径（包含参考关节轨迹和刚体位姿）")
    parser.add_argument("--xml_path", type=str, required=True,
                        help="MuJoCo XML 机器人模型文件路径")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="ONNX 策略模型文件路径")
    parser.add_argument("--save_json", action="store_true",
                        help="将运动数据另存为 JSON 格式（用于调试）")
    parser.add_argument("--loop", action="store_true",
                        help="运动序列结束后循环播放")
    
    args = parser.parse_args()
    
    print(f"[INFO]: Robot: {args.robot}")
    print(f"[INFO]: Motion file: {args.motion_file}")
    print(f"[INFO]: XML path: {args.xml_path}")
    print(f"[INFO]: Policy path: {args.policy_path}")
    
    run_simulation(args.robot, args.motion_file, args.xml_path, args.policy_path, args.save_json, args.loop)


if __name__ == "__main__":
    main()
