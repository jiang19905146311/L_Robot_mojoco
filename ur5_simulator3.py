'''
ur5_simulation.ur5_simulator1 的 Docstring
双机器人加视觉
'''
import os
# 配置后端
os.environ['MUJOCO_GL'] = 'glfw'

import mujoco
import mujoco.viewer
import time
import numpy as np
import cv2
import math

# === 导入自定义模块 ===
import utils_robot
from utils_vision import VisionModule



def make_T(rot_mat, pos_vec):
    """将3x3旋转矩阵和3x1平移向量组合成4x4齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos_vec
    return T

def get_body_T(data, body_name):
    """获取指定 body 在世界坐标系下的 4x4 变换矩阵"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = data.xpos[body_id]
    mat = data.xmat[body_id].reshape(3, 3)
    return make_T(mat, pos)

def get_camera_T(data, cam_name):
    """获取 camera 在世界坐标系下的 4x4 变换矩阵"""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    pos = data.cam_xpos[cam_id]
    mat = data.cam_xmat[cam_id].reshape(3, 3)
    return make_T(mat, pos)

# 1. 加载模型
xml_path = "/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/scene_dualcalib.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 步进一次以计算初始运动学
mujoco.mj_kinematics(model, data)

# ==========================================
# 2. 提取 Ground Truth (真值 X, Y, Z)
# ==========================================
# O1, O2: Base frames
T_W_O1 = get_body_T(data, "base")
T_W_O2 = get_body_T(data, "base_2")

# E1, E2: Flange frames
T_W_E1 = get_body_T(data, "wrist_3_link")
T_W_E2 = get_body_T(data, "wrist_3_link_2")

# S, T: Sensor and Target frames
T_W_S = get_camera_T(data, "wrist_camera")
T_W_T = get_body_T(data, "checkerboard_attached")

# 计算真实的 X, Y, Z (公式: Frame_parent^-1 * Frame_child)
# Y: O1 -> O2  (Base to Base)
Y_gt = np.linalg.inv(T_W_O1) @ T_W_O2
# X: E1 -> S   (Flange to Sensor)
X_gt = np.linalg.inv(T_W_E1) @ T_W_S
# Z: E2 -> T   (Flange to Target)
Z_gt = np.linalg.inv(T_W_E2) @ T_W_T

print("=== Ground Truth ===")
print("Y_gt (Base to Base):\n", np.round(Y_gt, 4))
print("X_gt (Hand to Eye):\n", np.round(X_gt, 4))
print("Z_gt (Flange to Target):\n", np.round(Z_gt, 4))

# ==========================================
# 3. 随机采样姿态并生成 A_i, B_i, C_i 数据集
# ==========================================
num_samples = 150  # 论文中建议 100-300
A_list, B_list, C_list = [], [], []

print("\n开始生成合法的观测数据...")
valid_count = 0
attempts = 0

while valid_count < num_samples:
    attempts += 1
    
    # 随机生成两个机械臂的关节角 (UR5 关节限制范围大约在 -pi 到 pi)
    # 给定一个比较合理的采样空间，避免全部都是自碰撞
    qpos_rand = np.random.uniform(-1.5, 1.5, model.nq)
    
    # 针对 shoulder_pan 可以扩大一点范围，让它们更容易面对面
    qpos_rand[0] += -1.57 # Robot 1 base
    qpos_rand[6] += -1.57 # Robot 2 base
    
    data.qpos[:] = qpos_rand
    mujoco.mj_kinematics(model, data)
    
    # 获取当前的 S 和 T 位姿
    T_W_S_current = get_camera_T(data, "wrist_camera")
    T_W_T_current = get_body_T(data, "checkerboard_attached")
    
    # --- 可见性约束检查 ---
    # 1. 距离检查 (假设标定板在相机 0.2m ~ 0.8m 之间最佳)
    p_S = T_W_S_current[:3, 3]
    p_T = T_W_T_current[:3, 3]
    dist = np.linalg.norm(p_T - p_S)
    if dist < 0.2 or dist > 1.0:
        continue
        
    # 2. 视角检查 (相机Z轴需要看向标定板面)
    # MuJoCo中相机默认看向 -Z 轴，标定板法向量假设为本地 +Z 轴
    cam_look_dir = -T_W_S_current[:3, 2] 
    target_normal = T_W_T_current[:3, 2]
    
    # 相机看向标定板的向量
    vec_S2T = (p_T - p_S) / dist
    
    # 相机视线必须对着标定板 (夹角小于 45 度)
    if np.dot(cam_look_dir, vec_S2T) < np.cos(np.deg2rad(45)):
        continue
        
    # 标定板法向量需要与相机视线大致相反 (夹角大于 135 度)
    if np.dot(target_normal, cam_look_dir) > np.cos(np.deg2rad(135)):
        continue
        
    # 如果通过了所有的检查，说明这是一个“有效”的标定姿态！
    valid_count += 1
    
    # 计算 A_i: O1 -> E1
    T_W_O1 = get_body_T(data, "base")
    T_W_E1 = get_body_T(data, "wrist_3_link")
    A_i = np.linalg.inv(T_W_O1) @ T_W_E1
    
    # 计算 C_i: O2 -> E2
    T_W_O2 = get_body_T(data, "base_2")
    T_W_E2 = get_body_T(data, "wrist_3_link_2")
    C_i = np.linalg.inv(T_W_O2) @ T_W_E2
    
    # 计算 B_i: S -> T (相机坐标系下的标定板位姿)
    B_i = np.linalg.inv(T_W_S_current) @ T_W_T_current
    
    # （可选：在这里你可以按照论文公式 44 给 A_i, B_i, C_i 加入噪声）
    
    A_list.append(A_i)
    B_list.append(B_i)
    C_list.append(C_i)

print(f"数据生成完毕！共尝试 {attempts} 次，成功采集 {num_samples} 组有效数据。")

# 可以将数据保存为 numpy 数组，供下一步算法计算使用
np.savez("calibration_data.npz", 
         A=np.array(A_list), 
         B=np.array(B_list), 
         C=np.array(C_list),
         X_gt=X_gt, Y_gt=Y_gt, Z_gt=Z_gt)