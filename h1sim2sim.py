"""H1 机器人专用 Sim-to-Sim 仿真验证脚本。

将 ONNX 策略模型部署到 MuJoCo 物理仿真器中，通过回放预录制的参考动作（.npz），
驱动策略网络实时推理并输出关节力矩，在仿真环境中验证策略的动作追踪效果。

使用示例:
    # 基本用法:
    python h1sim2sim.py --motion_file motions/merged_walk1.npz \
        --xml_path assets/unitree_g1_mjcf/h1sum.xml \
        --policy_path models/2026-06-09_13-07-05_h1_walk.onnx

    # 循环播放 + 自定义控制分频:
    python h1sim2sim.py --motion_file motions/merged_walk1.npz \
        --xml_path assets/unitree_g1_mjcf/h1sum.xml \
        --policy_path models/2026-06-09_13-07-05_h1_walk.onnx --loop --decimation 4
"""

from __future__ import annotations

import argparse
import json
import time
import os

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime

# ============================================================================
# H1 机器人固定参数
# ============================================================================
SIMULATION_DURATION = 300.0   # 仿真总时长（秒）
SIMULATION_DT = 0.005         # 物理仿真步长（秒），与 Isaac Lab 训练保持一致
NUM_ACTIONS = 19              # H1 受控关节数
NUM_OBS = 110                 # 观测向量维度: 38(指令) + 3(锚点位置) + 6(锚点朝向)
                              #   + 3(线速度) + 3(角速度) + 19(关节位置) + 19(关节速度) + 19(上一步动作)
REFERENCE_BODY = "pelvis"     # 参考刚体（锚点）
INIT_PELVIS_Z = 1.05          # 初始骨盆离地高度（米），对齐运动数据起始高度
DEFAULT_DECIMATION = 4        # 默认控制分频（每 4 个物理步 = 0.02s 推理一次，即 50Hz）


# ============================================================================
# 数学工具函数
# ============================================================================

def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    """将四元数转换为旋转矩阵。格式 (w, x, y, z)，标量在前。"""
    r, i, j, k = np.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_obs(data):
    """从 MuJoCo 仿真数据中提取机器人基础状态观测。

    返回: (qpos, dq, quat, v, omega, gvec, state_tau)
    """
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    # 优先从 IMU 传感器读取，回退到 free joint
    try:
        quat = data.sensor("orientation").data[[0, 1, 2, 3]].astype(np.double)
        omega = data.sensor("angular-velocity").data.astype(np.double)
    except:
        quat = data.qpos[3:7].astype(np.double)
        omega = data.qvel[3:6].astype(np.double)
    # 世界速度 → 机器人局部坐标
    rotm = np.zeros(9)
    mujoco.mju_quat2Mat(rotm, quat)
    rotm = rotm.reshape((3, 3))
    v = (rotm.T @ data.qvel[:3]).astype(np.double)
    gvec = (rotm.T @ np.array([0.0, 0.0, -1.0])).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)
    return (qpos, dq, quat, v, omega, gvec, state_tau)


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法（Hamilton 积）。"""
    if q1.shape != q2.shape:
        raise ValueError(f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}.")
    shape = q1.shape
    q1, q2 = q1.reshape(-1, 4), q2.reshape(-1, 4)
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
    """四元数共轭。"""
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)


def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """四元数的逆。"""
    return quat_conjugate_np(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), a_min=eps, a_max=None)


def quat_rotate_inverse_np(q, v):
    """用四元数的逆旋转来变换向量。v_local = R(q)^T * v"""
    q_w, q_vec = q[0], q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def subtract_frame_transforms_np(t01, q01, t02, q02):
    """计算帧 2 相对于帧 1 的变换（在帧 1 局部坐标系中）。

    返回: (t_12, q_12) — 相对位置和相对四元数
    """
    q01_inv = quat_inv_np(q01)
    q_12 = quat_mul_np(q01_inv, q02)
    t_12 = quat_rotate_inverse_np(q01, t02 - t01)
    return t_12, q_12


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD 位置控制器：τ = Kp*(q_tgt - q_cur) + Kd*(dq_tgt - dq_cur)"""
    return (target_q - q) * kp + (target_dq - dq) * kd


# ============================================================================
# ONNX 元数据解析
# ============================================================================

def parse_str_list(val):
    """解析字符串列表（JSON 或逗号分隔格式）。"""
    if not val: return []
    try:
        parsed = json.loads(val.replace("'", '"'))
        if isinstance(parsed, list): return parsed
    except: pass
    return [x.strip() for x in val.split(',')]


def parse_csv_list(val, dtype=float):
    """解析数值列表（JSON 或逗号分隔格式）。"""
    if not val: return []
    try:
        parsed = json.loads(val.replace("'", '"'))
        if isinstance(parsed, list):
            return [dtype(x) if isinstance(x, (int, float, str)) else x for x in parsed]
    except: pass
    return [dtype(x.strip()) for x in val.split(',')]


# ============================================================================
# 观测向量构造函数（H1 与 G1 布局一致）
# ============================================================================

def create_observation(obs, motioninput, v, omega, qpos_seq, qvel_seq, action_buffer,
                       default_joint_pos, robot_anchor_pos, robot_anchor_quat,
                       motion_anchor_pos, motion_anchor_quat):
    """构造 H1 观测向量（110 维），与训练环境完全对齐。

    布局:
      [0:38]   command             = concat(目标关节位置, 目标关节速度)  19×2=38
      [38:41]  motion_anchor_pos_b = 运动锚点在机器人局部坐标系中的位置
      [41:47]  motion_anchor_ori_b = 相对朝向的 6D 旋转表示
      [47:50]  base_lin_vel        = 基座线速度（局部坐标系）
      [50:53]  base_ang_vel        = 基座角速度（局部坐标系）
      [53:72]  joint_pos           = 关节位置偏差（当前 - 默认）
      [72:91]  joint_vel           = 关节速度
      [91:110] actions             = 上一步策略输出
    """
    cmd = motioninput if len(motioninput) == len(qpos_seq) * 2 else np.zeros(len(qpos_seq) * 2)
    anchor_pos_b, anchor_quat_b = subtract_frame_transforms_np(
        robot_anchor_pos, robot_anchor_quat, motion_anchor_pos, motion_anchor_quat
    )
    mat = matrix_from_quat(anchor_quat_b)
    anchor_ori_b = mat[..., :2].reshape(-1)
    obs_list = [
        cmd,
        anchor_pos_b,
        anchor_ori_b,
        v,
        omega,
        qpos_seq - default_joint_pos,
        qvel_seq,
        action_buffer,
    ]
    obs_array = np.concatenate(obs_list).astype(np.float32)
    if len(obs) < len(obs_array):
        obs = np.zeros(len(obs_array), dtype=np.float32)
    obs[:len(obs_array)] = obs_array
    return obs


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

    依次尝试:
      1. ONNX 元数据 body_names 筛选后的刚体集合
      2. MuJoCo 模型所有刚体（不含 world body）
      3. MuJoCo 模型所有刚体（含 world body）
      4. 后备索引
    """
    if anchor_body_name and metadata_body_names and motion_body_count == len(metadata_body_names):
        if anchor_body_name in metadata_body_names:
            return metadata_body_names.index(anchor_body_name)
    if anchor_body_id >= 0 and motion_body_count == model_body_count - 1:
        return anchor_body_id - 1
    if anchor_body_id >= 0 and motion_body_count == model_body_count:
        return anchor_body_id
    return fallback_index


# ============================================================================
# 主仿真循环
# ============================================================================

def run_simulation(motion_file: str, xml_path: str, policy_path: str,
                   control_decimation: int = DEFAULT_DECIMATION,
                   save_json: bool = False, loop: bool = False):
    """运行 H1 Sim-to-Sim 仿真验证。

    参数:
        motion_file:       运动数据文件路径（.npz 格式）
        xml_path:          MuJoCo XML 模型路径
        policy_path:       ONNX 策略文件路径
        control_decimation: 控制分频（每 N 个物理步执行一次策略推理）
        save_json:         是否将运动数据另存为 JSON
        loop:              是否循环播放
    """
    # ========================================================================
    # 第一步：加载运动数据
    # ========================================================================
    try:
        motion = np.load(motion_file)
        motionpos = motion["body_pos_w"]
        motionquat = motion["body_quat_w"]
        motioninputpos = motion["joint_pos"]
        motioninputvel = motion["joint_vel"]
        num_frames = min(motioninputpos.shape[0], motioninputvel.shape[0],
                         motionpos.shape[0], motionquat.shape[0])
        print(f"[INFO]: Loaded motion: {num_frames} frames, "
              f"{motioninputpos.shape[1]} joints, {motionpos.shape[1]} bodies")
    except Exception as e:
        print(f"[WARNING]: Motion file unreadable ({e}), falling back to dummy sequence.")
        num_frames = 1000
        motionpos = np.zeros((num_frames, 1, 3))
        motionquat = np.zeros((num_frames, 1, 4))
        motioninputpos = np.zeros((num_frames, NUM_ACTIONS))
        motioninputvel = np.zeros((num_frames, NUM_ACTIONS))

    def frame_idx(t):
        """根据当前时间步获取运动帧索引。"""
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1

    if save_json:
        motion_dict = {
            "body_pos_w": motionpos.tolist(),
            "body_quat_w": motionquat.tolist(),
            "joint_pos": motioninputpos.tolist(),
            "joint_vel": motioninputvel.tolist(),
        }
        json_path = os.path.splitext(motion_file)[0] + ".json"
        with open(json_path, 'w') as f:
            json.dump(motion_dict, f, indent=2)
        print(f"[INFO]: Motion data saved to: {json_path}")

    # ========================================================================
    # 第二步：加载 ONNX 策略模型并提取元数据
    # ========================================================================
    model = onnx.load(policy_path)
    joint_seq = None
    joint_pos_array_seq = None
    stiffness_array_seq = None
    damping_array_seq = None
    action_scale = None
    anchor_body_name = None
    metadata_body_names = None

    for prop in model.metadata_props:
        if prop.key == "joint_names":
            joint_seq = parse_str_list(prop.value)
        elif prop.key == "default_joint_pos":
            joint_pos_array_seq = np.array(parse_csv_list(prop.value))
        elif prop.key == "joint_stiffness":
            stiffness_array_seq = np.array(parse_csv_list(prop.value))
        elif prop.key == "joint_damping":
            damping_array_seq = np.array(parse_csv_list(prop.value))
        elif prop.key == "action_scale":
            action_scale = np.array(parse_csv_list(prop.value))
        elif prop.key == "anchor_body_name":
            anchor_body_name = prop.value
        elif prop.key == "body_names":
            metadata_body_names = parse_str_list(prop.value)

    if joint_seq is None:
        raise ValueError("ONNX 模型中未找到 joint_names 元数据，无法继续。")
    print(f"[INFO]: ONNX joints ({len(joint_seq)}): {joint_seq}")

    # ========================================================================
    # 第三步：加载 MuJoCo 物理模型
    # ========================================================================
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = SIMULATION_DT

    # ========================================================================
    # 第四步：建立关节名称映射（ONNX 顺序 ↔ XML 顺序）
    # ========================================================================
    # 从 XML 提取关节名（跳过 free joint），仅保留 ONNX 中也存在的关节
    xml_joint_names_full = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(1, m.njnt)]
    # 过滤掉 XML 中有但 ONNX 中无的关节（如 not_use_joint）
    joint_xml = [j for j in xml_joint_names_full if j in joint_seq]
    skipped = set(xml_joint_names_full) - set(joint_xml)
    if skipped:
        print(f"[INFO]: Skipped XML-only joints (not in ONNX): {skipped}")
    print(f"[INFO]: Mapped joints: {len(joint_xml)} (ONNX) ↔ {len(joint_xml)} (XML)")

    # 将 ONNX 顺序的参数重映射到 XML 顺序
    joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(j)] for j in joint_xml])

    if stiffness_array_seq is None or len(stiffness_array_seq) == 0:
        print("[WARNING]: Missing joint_stiffness in ONNX metadata, using zero PD gains.")
        stiffness_array = np.zeros(len(joint_xml))
        damping_array = np.zeros(len(joint_xml))
    else:
        stiffness_array = np.array([stiffness_array_seq[joint_seq.index(j)] for j in joint_xml])
        damping_array = np.array([damping_array_seq[joint_seq.index(j)] for j in joint_xml])

    if action_scale is None or len(action_scale) == 0:
        action_scale = np.ones(len(joint_xml))
    print(f"[INFO]: action_scale: {[f'{s:.3f}' for s in action_scale]}")

    # ========================================================================
    # 第五步：解析锚点刚体索引
    # ========================================================================
    reference_body = anchor_body_name or REFERENCE_BODY
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, reference_body)
    if body_id == -1:
        raise ValueError(f"锚点刚体 '{reference_body}' 在模型 {xml_path} 中未找到。")

    motion_body_idx = resolve_motion_anchor_index(
        motion_body_count=motionpos.shape[1],
        model_body_count=m.nbody,
        anchor_body_id=body_id,
        anchor_body_name=reference_body,
        metadata_body_names=metadata_body_names,
        fallback_index=0,
    )
    print(f"[INFO]: Anchor body: '{reference_body}' (XML id={body_id}, motion idx={motion_body_idx})")

    # ========================================================================
    # 第六步：初始化仿真状态
    # ========================================================================
    num_actions = len(joint_seq)
    num_obs = NUM_OBS
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    policy = onnxruntime.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
    input_shape = policy.get_inputs()[0].shape
    print(f"[INFO]: ONNX expects obs shape: {input_shape}")

    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0

    # 初始运动指令
    minp = motioninputpos[frame_idx(timestep), :]
    minv = motioninputvel[frame_idx(timestep), :]
    if len(minp) > num_actions: minp = minp[:num_actions]
    if len(minv) > num_actions: minv = minv[:num_actions]
    motioninput = np.concatenate((minp, minv), axis=0)

    # 设置初始状态
    target_dof_pos = joint_pos_array.copy()
    d.qpos[2] = INIT_PELVIS_Z
    if len(d.qpos) > 7:
        d.qpos[7:7 + len(joint_xml)] = joint_pos_array
    d.qvel[:] = 0.0
    if m.nu > 0:
        d.ctrl[:] = 0.0
    d.qfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)

    # ========================================================================
    # 第七步：主仿真循环
    # ========================================================================
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < SIMULATION_DURATION:
            step_start = time.time()

            qpos, dq, quat, v, omega, gvec, state_tau = get_obs(d)

            # ------ 策略推理（每 control_decimation 步执行一次）------
            if counter % control_decimation == 0:
                idx = frame_idx(timestep)

                # 当前帧运动指令
                minp = motioninputpos[idx, :]
                minv = motioninputvel[idx, :]
                if len(minp) > num_actions: minp = minp[:num_actions]
                if len(minv) > num_actions: minv = minv[:num_actions]
                motioninput = np.concatenate((minp, minv), axis=0)

                # 关节状态：XML 顺序 → ONNX 顺序
                qpos_xml = d.qpos[7:7 + len(joint_xml)]
                qpos_seq = np.array([qpos_xml[joint_xml.index(j)] for j in joint_seq])
                qvel_xml = d.qvel[6:6 + len(joint_xml)]
                qvel_seq = np.array([qvel_xml[joint_xml.index(j)] for j in joint_seq])

                # 锚点位姿
                robot_anchor_pos = d.xpos[body_id].copy()
                robot_anchor_quat = d.xquat[body_id].copy()
                motion_anchor_pos = motionpos[idx, motion_body_idx, :].copy()
                motion_anchor_quat = motionquat[idx, motion_body_idx, :].copy()

                # 构造观测 → ONNX 推理
                obs = create_observation(
                    obs, motioninput, v, omega, qpos_seq, qvel_seq, action_buffer,
                    joint_pos_array_seq, robot_anchor_pos, robot_anchor_quat,
                    motion_anchor_pos, motion_anchor_quat,
                )

                # 维度对齐
                target_dim = input_shape[1]
                if obs.shape[0] < target_dim:
                    obs = np.pad(obs, (0, target_dim - obs.shape[0]))
                elif obs.shape[0] > target_dim:
                    obs = obs[:target_dim]

                obs_tensor = np.expand_dims(obs, axis=0)
                action_out = policy.run(['actions'], {
                    policy.get_inputs()[0].name: obs_tensor,
                    policy.get_inputs()[1].name: np.array([[frame_idx(timestep)]], dtype=np.float32),
                })[0]

                # 策略输出 → 目标关节位置（ONNX 顺序 → XML 顺序）
                action_array = np.asarray(action_out).reshape(-1)
                action_buffer = action_array.copy()
                target_dof_seq = action_array * action_scale + joint_pos_array_seq
                target_dof_pos = np.array([target_dof_seq[joint_seq.index(j)] for j in joint_xml])

                if loop or timestep + 1 < num_frames:
                    timestep += 1

            # ------ PD 控制 + 物理步进 ------
            if np.any(stiffness_array > 0):
                tau = pd_control(target_dof_pos, d.qpos[7:7 + len(joint_xml)],
                                 stiffness_array, np.zeros(len(joint_xml)),
                                 d.qvel[6:6 + len(joint_xml)], damping_array)
                if m.nu == len(joint_xml):
                    d.ctrl[:] = tau
                else:
                    d.qfrc_applied[:] = 0.0
                    d.qfrc_applied[6:6 + len(joint_xml)] = tau
            else:
                if m.nu == len(joint_xml):
                    d.ctrl[:] = target_dof_pos
                else:
                    d.qfrc_applied[:] = 0.0

            mujoco.mj_step(m, d)
            counter += 1
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="H1 机器人专用 Sim-to-Sim 仿真验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python h1sim2sim.py --motion_file motions/merged_walk1.npz \\
      --xml_path assets/unitree_g1_mjcf/h1sum.xml \\
      --policy_path models/2026-06-09_13-07-05_h1_walk.onnx

  python h1sim2sim.py --motion_file motions/merged_walk1.npz \\
      --xml_path assets/unitree_g1_mjcf/h1sum.xml \\
      --policy_path models/2026-06-09_13-07-05_h1_walk.onnx --loop --decimation 4
        """,
    )
    parser.add_argument("--motion_file", type=str, required=True,
                        help="运动数据 NPZ 文件路径")
    parser.add_argument("--xml_path", type=str, required=True,
                        help="MuJoCo XML 机器人模型路径")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="ONNX 策略模型路径")
    parser.add_argument("--decimation", type=int, default=DEFAULT_DECIMATION,
                        help=f"控制分频，每 N 个物理步推理一次（默认: {DEFAULT_DECIMATION}）")
    parser.add_argument("--save_json", action="store_true",
                        help="将运动数据另存为 JSON")
    parser.add_argument("--loop", action="store_true",
                        help="运动序列结束后循环播放")

    args = parser.parse_args()
    print(f"[INFO]: Motion file: {args.motion_file}")
    print(f"[INFO]: XML path: {args.xml_path}")
    print(f"[INFO]: Policy path: {args.policy_path}")
    print(f"[INFO]: Decimation: {args.decimation}")
    print(f"[INFO]: Loop: {args.loop}")

    run_simulation(args.motion_file, args.xml_path, args.policy_path,
                   control_decimation=args.decimation,
                   save_json=args.save_json, loop=args.loop)


if __name__ == "__main__":
    main()
