'''
utils_robot.py 的 Docstring
机器人相关的工具函数，包括模型加载、状态重置、逆运动学求解以及工作空间点云生成等功能。
'''
import mujoco
import numpy as np

# 缓存关节索引，避免重复查询
_ROBOT_JOINT_INDICES = None 

def load_model(xml_path):
    """加载模型并返回 model, data"""
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        return model, data
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def reset_robot(model, data, gripper_open_val):
    """重置机器人到初始姿态"""
    mujoco.mj_resetData(model, data)
    # UR5 标准 Home 姿态
    home_qpos = [-1.57, -1.57, 1.57, -1.57, -1.57, 0]
    data.qpos[:6] = home_qpos
    data.ctrl[:6] = home_qpos
    data.ctrl[6] = gripper_open_val
    data.ctrl[7] = gripper_open_val
    
    # 设置方块位置 (初始化一下，防止方块飞了)
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if obj_id != -1:
        qpos_adr = model.jnt_qposadr[obj_id]
        data.qpos[qpos_adr:qpos_adr+3] = [0.5, 0.0, 0.03]
        data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
        
    mujoco.mj_forward(model, data)
    print(">>> 机器人已复位")

def get_joint_indices(model):
    """(内部使用) 获取关节索引"""
    names = [f"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    indices = []
    for name in names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id != -1:
            indices.append(model.jnt_dofadr[j_id])
    return np.array(indices, dtype=np.int32)

def solve_ik(model, data, target_pos, target_quat, damping=1e-2):
    """
    计算 6D 逆运动学
    返回: delta_q (关节速度/增量)
    """
    global _ROBOT_JOINT_INDICES
    if _ROBOT_JOINT_INDICES is None:
        _ROBOT_JOINT_INDICES = get_joint_indices(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    
    # 1. 计算误差 (位置 + 姿态)
    current_pos = data.site(site_id).xpos
    err_pos = target_pos - current_pos

    current_mat = data.site(site_id).xmat.reshape(3, 3)
    current_quat = np.zeros(4)
    mujoco.mju_mat2Quat(current_quat, current_mat.flatten())
    
    neg_current_quat = current_quat.copy()
    neg_current_quat[1:] *= -1
    err_quat = np.zeros(4)
    mujoco.mju_mulQuat(err_quat, target_quat, neg_current_quat)
    
    err_rot = np.zeros(3)
    mujoco.mju_quat2Vel(err_rot, err_quat, 1)
    
    error = np.hstack([err_pos, err_rot])
    
    # 2. 计算雅可比
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    
    J_full = np.vstack([jacp, jacr])
    J_ur5 = J_full[:, _ROBOT_JOINT_INDICES]
    
    # 3. 阻尼最小二乘求解
    H = J_ur5.T @ J_ur5 + damping * np.eye(6)
    delta_q = np.linalg.solve(H, J_ur5.T @ error)
    
    return delta_q