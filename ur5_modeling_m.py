'''
UR5机械臂标定与误差分析
本脚本实现了基于 POE（Product of Exponentials）方法的 UR5 机械臂标定。通过优化关节参数和末端执行器零位矩阵的误差，我们可以显著提升机械臂的运动学精度。
主要步骤包括：
1. 从 MuJoCo 模型中提取名义的 POE 参数（零位矩阵 M 和 6 个关节的旋量）。
2. 定义一个带误差参数的正运动学函数，能够根据当前的关节角和误差参数计算末端执行器的位姿。
3. 构建一个优化目标函数，计算所有样本的预测位姿与真实测量位姿之间的残差。
4. 使用 SciPy 的 Levenberg-Marquardt 算法进行非线性优化，求解出最优的误差参数。
5. 输出优化前后的误差统计，并保存标定结果。
请确保在运行此脚本前，已经完成了数据采集并将 q_data.npy 和 tcp_data.npy 文件放在当前目录下。
'''

import numpy as np
import mujoco
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares

# ==========================================
# 1. 配置参数
# ==========================================
XML_PATH = "/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/ur5.xml"
SITE_NAME = "tcp"
DATA_Q = "2026.05.16calibration_q.npy"
DATA_TCP = "2026.05.16calibration_tcp.npy"

# ==========================================
# 2. 基础数学与 POE 工具函数
# ==========================================
def skew_symmetric(w):
    """向量转反对称矩阵 (so(3))"""
    return np.array([
        [    0, -w[2],  w[1]],
        [ w[2],     0, -w[0]],
        [-w[1],  w[0],     0]
    ])

def twist_to_matrix(w, v):
    """旋量转 4x4 齐次变换矩阵的指数映射输入 (se(3))"""
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew_symmetric(w)
    mat[:3, 3] = v
    return mat

def get_nominal_poe_params():
    """从 MuJoCo 提取绝对理想的名义参数 (包含之前修复的全局坐标系逻辑)"""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    original_qpos = data.qpos.copy()
    data.qpos[:6] = 0
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    M = np.eye(4)
    M[:3, :3] = data.site(site_id).xmat.reshape(3, 3)
    M[:3, 3] = data.site(site_id).xpos

    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]

    screws = []
    for name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        
        # 获取全局旋转轴
        body_id = model.jnt_bodyid[j_id]
        body_xmat = data.xmat[body_id].reshape(3, 3)
        w_local = model.jnt_axis[j_id].copy() 
        w_global = body_xmat @ w_local
        
        q_anchor = data.xanchor[j_id].copy()
        v = -np.cross(w_global, q_anchor)
        screws.append((w_global, v))

    return M, screws

# ==========================================
# 3. 标定核心：带误差参数的 POE 正运动学
# ==========================================
def fk_calibrated(q, nominal_screws, nominal_M, params):
    """
    params 长度为 42：
    [0:36] 分别为 6 个关节的旋量微小偏差 delta_xi (每个关节6个值)
    [36:42] 为零位矩阵 M 的微小位姿偏差 delta_M
    """
    T = np.eye(4)
    
    # 遍历计算每个关节的指数积
    for i in range(6):
        # 提取该关节的 6 维误差参数 (dwx, dwy, dwz, dvx, dvy, dvz)
        d_xi = params[i*6 : i*6+6]
        # 误差构成的微小位移矩阵 exp(delta_xi)
        T_err = expm(twist_to_matrix(d_xi[:3], d_xi[3:]))
        
        # 提取名义参数
        w, v = nominal_screws[i]
        # 名义运动矩阵 exp(xi * q)
        T_nom = expm(twist_to_matrix(w, v) * q[i])
        
        # 真正的关节变换 = 误差变换 * 名义变换
        T = T @ T_err @ T_nom
        
    # 处理 TCP 零位矩阵 M 的误差
    d_M = params[36:42]
    T_M_err = expm(twist_to_matrix(d_M[:3], d_M[3:]))
    
    # 将 M 的误差叠加到末端
    T = T @ nominal_M @ T_M_err
    return T

# ==========================================
# 4. 优化目标函数 (Cost Function)
# ==========================================
def cost_function(params, q_data, tcp_data, nominal_screws, nominal_M):
    """计算所有样本的预测值与真实值之间的残差"""
    residuals = []
    
    for i in range(len(q_data)):
        q = q_data[i]
        tcp_meas = tcp_data[i]
        
        # 真实的 TCP 数据
        pos_meas = tcp_meas[:3]
        r_meas = R_scipy.from_rotvec(tcp_meas[3:6])
        
        # 带入当前迭代参数计算的预测 TCP 数据
        T_calc = fk_calibrated(q, nominal_screws, nominal_M, params)
        pos_calc = T_calc[:3, 3]
        r_calc = R_scipy.from_matrix(T_calc[:3, :3])
        
        # 1. 位置误差 (单位：米)
        pos_err = pos_calc - pos_meas
        
        # 2. 姿态误差 (单位：弧度，使用 SO(3) 对数映射)
        ori_err = (r_meas * r_calc.inv()).as_rotvec()
        
        # 将误差加入残差数组 (优化器会去最小化这个数组中所有元素的平方和)
        residuals.extend(pos_err)
        residuals.extend(ori_err)
        
    return np.array(residuals)

# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    print("1. Loading nominal parameters from MuJoCo...")
    nominal_M, nominal_screws = get_nominal_poe_params()
    
    print("2. Loading collected real robot data...")
    # 请确保这两个文件与当前脚本在同一目录下
    q_data = np.load(DATA_Q)
    tcp_data = np.load(DATA_TCP)
    num_samples = len(q_data)
    print(f"   Loaded {num_samples} samples.")

    # 计算标定前的初始误差
    initial_params = np.zeros(42)  # 初始认为没有误差
    initial_residuals = cost_function(initial_params, q_data, tcp_data, nominal_screws, nominal_M)
    
    # 每 6 个数据对应一个位姿的 [x, y, z, rx, ry, rz] 误差
    initial_pos_errs = np.linalg.norm(initial_residuals.reshape(-1, 6)[:, :3], axis=1) * 1000.0  # 转换为 mm
    initial_ori_errs = np.linalg.norm(initial_residuals.reshape(-1, 6)[:, 3:6], axis=1) * (180.0/np.pi)  # 转换为 degree
    
    print(f"\n--- BEFORE CALIBRATION (Nominal Model) ---")
    print(f"Mean Position Error: {np.mean(initial_pos_errs):.4f} mm  (Max: {np.max(initial_pos_errs):.4f} mm)")
    print(f"Mean Orientation Error: {np.mean(initial_ori_errs):.4f} deg (Max: {np.max(initial_ori_errs):.4f} deg)")

    print("\n3. Running Levenberg-Marquardt Optimization... (This might take a few seconds)")
    
    # 核心：调用 SciPy 的非线性最小二乘法优化器
    result = least_squares(
        fun=cost_function,
        x0=initial_params,
        args=(q_data, tcp_data, nominal_screws, nominal_M),
        method='lm',     # Levenberg-Marquardt 算法 (特别适合机器人运动学标定)
        verbose=1        # 显示优化进度
    )
    
    optimized_params = result.x

    # 计算标定后的误差
    final_residuals = cost_function(optimized_params, q_data, tcp_data, nominal_screws, nominal_M)
    final_pos_errs = np.linalg.norm(final_residuals.reshape(-1, 6)[:, :3], axis=1) * 1000.0
    final_ori_errs = np.linalg.norm(final_residuals.reshape(-1, 6)[:, 3:6], axis=1) * (180.0/np.pi)
    
    print(f"\n--- AFTER CALIBRATION (Optimized Model) ---")
    print(f"Mean Position Error: {np.mean(final_pos_errs):.6f} mm  (Max: {np.max(final_pos_errs):.6f} mm)")
    print(f"Mean Orientation Error: {np.mean(final_ori_errs):.6f} deg (Max: {np.max(final_ori_errs):.6f} deg)")

    if result.success:
        print("\n🎉 Calibration SUCCESSFUL!")
        # 保存标定出的补偿参数
        np.save("optimized_error_params.npy", optimized_params)
        print("Calibrated error parameters saved to 'optimized_error_params.npy'")
    else:
        print("\n⚠️ Calibration failed to converge.")