'''
UR5机械臂正向运动学验证，几何误差分析以及补偿 (数字孪生版)
需要保证UR5机械臂的控制器已经在运行，并且RTDE接口可用。
主要功能：
- 通过 RTDE 实时获取真实 UR5 机械臂的关节角度。
- 使用真实角度实时驱动 MuJoCo 仿真环境中的 UR5 模型（Digital Twin）。
- 提取 PoE 参数并在每个仿真步中计算末端位姿。
- 在 DearPyGui 界面中实时显示仿真TCP位置、PoE计算位置及误差曲线，验证算法正确性。
'''

import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
from collections import deque
from scipy.linalg import expm

import dearpygui.dearpygui as dpg
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation as R_scipy

from utils_robot import KinematicsUtils


# ============================================
# 配置
# ============================================

XML_PATH = "/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/ur5.xml"
SITE_NAME = "tcp"
ROBOT_IP = "192.168.134.100"

# ============================================
# POE 参数提取
# ============================================

def get_poe_params(model, data):

    # 保存原始关节角，将所有关节归零（获取零位）
    original_qpos = data.qpos.copy()
    data.qpos[:6] = 0

    # 前向动力学计算，更新位姿
    mujoco.mj_forward(model, data)

    # 获取TCP末端ID，计算零位位姿矩阵M
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        SITE_NAME
    )

    M = np.eye(4)
    M[:3, :3] = data.site(site_id).xmat.reshape(3, 3)
    M[:3, 3] = data.site(site_id).xpos

    # 遍历6个UR5关节，提取每个关节的旋量(w, v)
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    screws = []

    for name in joint_names:

        j_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name
        )

        # 获取该关节所属的 body ID
        body_id = model.jnt_bodyid[j_id]
        
        # 获取该 body 的全局旋转矩阵 (3x3)
        body_xmat = data.xmat[body_id].reshape(3, 3)
        
        # 将局部旋转轴转换为全局旋转轴
        w_local = model.jnt_axis[j_id].copy() 
        w_global = body_xmat @ w_local
        
        # 锚点 xanchor 已经是全局坐标
        q_anchor = data.xanchor[j_id].copy()
        
        # 计算线速度 v
        v = -np.cross(w_global, q_anchor)
        screws.append((w_global, v))

    # 恢复原始关节角，返回参数
    data.qpos[:] = original_qpos
    mujoco.mj_forward(model, data)

    return M, screws

# ============================================
# 误差补偿的 POE 正运动学
# ============================================

def skew_symmetric(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def twist_to_matrix(w, v):
    mat = np.zeros((4, 4))
    mat[:3, :3] = skew_symmetric(w)
    mat[:3, 3] = v
    return mat

def fk_calibrated(q, nominal_screws, nominal_M, params):
    """带误差补偿的 POE 计算"""
    T = np.eye(4)
    for i in range(6):
        d_xi = params[i*6 : i*6+6]
        T_err = expm(twist_to_matrix(d_xi[:3], d_xi[3:]))
        w, v = nominal_screws[i]
        T_nom = expm(twist_to_matrix(w, v) * q[i])
        T = T @ T_err @ T_nom
    d_M = params[36:42]
    T_M_err = expm(twist_to_matrix(d_M[:3], d_M[3:]))
    T = T @ nominal_M @ T_M_err
    return T



# ============================================
# GUI 初始化
# ============================================

dpg.create_context()  # 初始化GUI上下文

pos_error_buffer = deque(maxlen=200)  # 误差缓存队列（最多存200个数据，自动丢弃旧数据）
ori_error_buffer = deque(maxlen=200)  # 姿态误差缓存

# 创建主窗口
with dpg.window(label="Kinematics Monitor", width=600, height=900):

    # --- 关节角 ---
    # 显示实时关节角
    dpg.add_text("Real Joint Angles (rad)[mm]")
    dpg.add_text("", tag="real_q_rad_text", color=[100, 255, 100])

    dpg.add_text("Real Joint Angles (deg)[mm]")
    dpg.add_text("", tag="real_q_deg_text", color=[100, 255, 100])

    # 视觉分割线
    dpg.add_separator()

    # --- 位置对比 (x, y, z) ---
    dpg.add_text("--- Position (x, y, z) [mm] ---", color=[200, 200, 255])

    # 显示真实控制器读取的 TCP
    dpg.add_text("RTDE Controller TCP Position (Ground Truth)[mm]")
    dpg.add_text("", tag="rtde_pos", color=[255, 200, 100]) 

    # 显示mujoco位置
    dpg.add_text("MuJoCo Digital Twin TCP Position[mm]")
    dpg.add_text("", tag="mujoco_pos")  # tag：唯一标识，用于后续更新数值

    # 显示POE计算位置
    dpg.add_text("POE Computed TCP Position[mm]")
    dpg.add_text("", tag="poe_pos")
    
    dpg.add_separator()

    # --- 姿态对比 (Euler RPY) ---
    dpg.add_text("--- Orientation (Euler XYZ) [deg] ---", color=[200, 200, 255])

    dpg.add_text("RTDE Controller TCP Orientation")
    dpg.add_text("", tag="rtde_ori", color=[255, 200, 100]) 

    dpg.add_text("MuJoCo Digital Twin TCP Orientation")
    dpg.add_text("", tag="mujoco_ori")  

    dpg.add_text("POE Computed TCP Orientation")
    dpg.add_text("", tag="poe_ori")

    dpg.add_separator()

    # --- 误差数值 ---

    dpg.add_text("Position Error (RTDE vs Mujoco)[mm]")
    dpg.add_text("", tag="Mujoco_pos_error_text", color=[255, 100, 100])
    dpg.add_text("Position Error (RTDE vs POE)[mm]")
    dpg.add_text("", tag="POE_pos_error_text", color=[255, 100, 100])
    dpg.add_text("", tag="POE_error1", color=[255, 200, 100]) 
 

    dpg.add_text("Orientation Error (RTDE vs Mujoco) [deg]")
    dpg.add_text("", tag="Mujoco_ori_error_text", color=[255, 100, 100])
    dpg.add_text("Orientation Error (RTDE vs POE) [deg]")
    dpg.add_text("", tag="POE_ori_error_text", color=[255, 100, 100])
    dpg.add_text("", tag="POE_error2", color=[255, 200, 100]) 
    
    
    # 误差曲线图
    with dpg.group(horizontal=True):
        with dpg.plot(label="Position Error", height=200, width=290):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="x_axis_pos")
            dpg.add_plot_axis(dpg.mvYAxis, label="Error(mm)", tag="y_axis_pos")
            dpg.add_line_series([], [], parent="y_axis_pos", tag="pos_error_curve")
            
        with dpg.plot(label="Orientation Error", height=200, width=290):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="x_axis_ori")
            dpg.add_plot_axis(dpg.mvYAxis, label="Error(deg)", tag="y_axis_ori")
            dpg.add_line_series([], [], parent="y_axis_ori", tag="ori_error_curve")

# 创建GUI视口
dpg.create_viewport(title="UR5 Monitor", width=620, height=450)
dpg.setup_dearpygui()
dpg.show_viewport()


# ============================================
# 主线程（MuJoCo & RTDE 主线程）
# ============================================

def mujoco_loop():
    # 加载UR5模型，初始化仿真数据
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 提取POE参数（零位矩阵M + 关节旋量）
    M, screws = get_poe_params(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)

    # =============== 加载优化好的补偿参数 ===============
    try:
        optimized_params = np.load("optimized_error_params.npy")
        print("Successfully loaded Calibrated Parameters!")
    except:
        print("Warning: optimized_error_params.npy not found, using zero errors.")
        optimized_params = np.zeros(42)


    # 连接真实的 UR5 机械臂
    print(f"Connecting to UR5 at {ROBOT_IP}...")
    try:
        rtde_r = RTDEReceiveInterface(ROBOT_IP)
        print("Connected successfully! Starting synchronization...")
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        return

    # 启动MuJoCo被动查看器（仅渲染，无交互控制）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():  # 仿真主循环
            step_start = time.time()

            # ================= 读取真实机器人数据 =================
            # 获取实际关节角度
            actual_q = rtde_r.getActualQ()
            # 获取实际TCP位姿 [x, y, z, rx, ry, rz]
            actual_tcp_pose = rtde_r.getActualTCPPose() 
            # 提取前三个元素作为真实的位置系数值 (毫米)
            rtde_pos = np.array(actual_tcp_pose[:3]) 
            # UR控制器的旋转是旋转向量 (Rotation Vector)
            rtde_rotvec = np.array(actual_tcp_pose[3:6]) 
            # 转换为 scipy 的 Rotation 对象
            r_rtde = R_scipy.from_rotvec(rtde_rotvec)

            # ================= 更新 MuJoCo 数字孪生 =================
            data.qpos[:6] = actual_q
            mujoco.mj_forward(model, data)  
            mujoco_pos = data.site(site_id).xpos.copy() 
            mujoco_xmat = data.site(site_id).xmat.reshape(3, 3)

            # MuJoCo 的旋转矩阵转 Rotation 对象
            r_mujoco = R_scipy.from_matrix(mujoco_xmat)
            
            # ================= PoE 算法计算 =================
            '''
            T_poe = KinematicsUtils.forward_kinematics_poe(actual_q, screws, M)
            poe_pos = T_poe[:3, 3] 
            poe_xmat = T_poe[:3, :3]

            # PoE 的旋转矩阵转 Rotation 对象
            r_poe = R_scipy.from_matrix(poe_xmat)
            '''
            # =============== 【核心修改】替换原来的 POE 算法 ===============
            # 现在换成带误差补偿的算法！
            T_poe = fk_calibrated(actual_q, screws, M, optimized_params)
            
            poe_pos = T_poe[:3, 3] 
            poe_xmat = T_poe[:3, :3]
            r_poe = R_scipy.from_matrix(poe_xmat)


            # ================= 计算误差 =================
            # --- 【位置误差 (mm)】 ---
            # 将米转换为毫米
            rtde_pos_mm = rtde_pos * 1000.0
            mujoco_pos_mm = mujoco_pos * 1000.0
            poe_pos_mm = poe_pos * 1000.0


            # 这里我们计算 RTDE真实读取值 与 你的POE算法 之间的误差
            Mujoco_pos_error = np.linalg.norm(rtde_pos_mm - mujoco_pos_mm)
            POE_pos_error = np.linalg.norm(rtde_pos_mm - poe_pos_mm)

            # 【姿态误差 (deg)】
            # 两个旋转矩阵的绝对差异 = r1 * r2.inv()，然后取其旋转角 magnitude
            Mujoco_ori_error_rad = (r_rtde * r_mujoco.inv()).magnitude()
            POE_ori_error_rad = (r_rtde * r_poe.inv()).magnitude()
            
            Mujoco_ori_error_deg = np.degrees(Mujoco_ori_error_rad)
            POE_ori_error_deg = np.degrees(POE_ori_error_rad)

            # ================= 更新GUI界面 =================
            q_rad_str = ", ".join([f"{rad:.3f}" for rad in actual_q])
            q_deg_str = ", ".join([f"{np.degrees(rad):.3f}" for rad in actual_q])
            dpg.set_value("real_q_rad_text", f"[{q_rad_str}]")
            dpg.set_value("real_q_deg_text", f"[{q_deg_str}]")

            # 格式化坐标输出方便肉眼对比
            rtde_str = f"[{rtde_pos_mm[0]:.5f}, {rtde_pos_mm[1]:.5f}, {rtde_pos_mm[2]:.5f}]"
            mujoco_str = f"[{mujoco_pos_mm[0]:.5f}, {mujoco_pos_mm[1]:.5f}, {mujoco_pos_mm[2]:.5f}]"
            poe_str = f"[{poe_pos_mm[0]:.5f}, {poe_pos_mm[1]:.5f}, {poe_pos_mm[2]:.5f}]"

            dpg.set_value("rtde_pos", rtde_str)
            dpg.set_value("mujoco_pos", mujoco_str)
            dpg.set_value("poe_pos", poe_str)
           
            # 更新姿态文本（为了肉眼好判断，转换为欧拉角 XYZ 显示，单位：度）
            rtde_euler = r_rtde.as_euler('xyz', degrees=True)
            mujoco_euler = r_mujoco.as_euler('xyz', degrees=True)
            poe_euler = r_poe.as_euler('xyz', degrees=True)

            dpg.set_value("rtde_ori", f"[{rtde_euler[0]:.3f}, {rtde_euler[1]:.3f}, {rtde_euler[2]:.3f}]")
            dpg.set_value("mujoco_ori", f"[{mujoco_euler[0]:.3f}, {mujoco_euler[1]:.3f}, {mujoco_euler[2]:.3f}]")
            dpg.set_value("poe_ori", f"[{poe_euler[0]:.3f}, {poe_euler[1]:.3f}, {poe_euler[2]:.3f}]")

            # 更新误差数值文本

            dpg.set_value("Mujoco_pos_error_text", f"{Mujoco_pos_error:.6f}")
            dpg.set_value("POE_pos_error_text", f"{POE_pos_error:.6f}")
            dpg.set_value("POE_error1", f"[{rtde_pos_mm[0] - poe_pos_mm[0]:.5f}, {rtde_pos_mm[1] - poe_pos_mm[1]:.5f}, {rtde_pos_mm[2] - poe_pos_mm[2]:.5f}]")  
            dpg.set_value("Mujoco_ori_error_text", f"{Mujoco_ori_error_deg:.6f}")
            dpg.set_value("POE_ori_error_text", f"{POE_ori_error_deg:.6f}")
            dpg.set_value("POE_error2", f"[{rtde_euler[0] - poe_euler[0]:.5f}, {rtde_euler[1] - poe_euler[1]:.5f}, {rtde_euler[2] - poe_euler[2]:.5f}]")  

            # 更新误差曲线
            pos_error_buffer.append(POE_pos_error)
            ori_error_buffer.append(POE_ori_error_deg)
            dpg.set_value("pos_error_curve", [list(range(len(pos_error_buffer))), list(pos_error_buffer)])
            dpg.set_value("ori_error_curve", [list(range(len(ori_error_buffer))), list(ori_error_buffer)])

            # 同步仿真器 + 控制刷新率（与仿真步长一致）
            viewer.sync()

            # 控制刷新率（尽量对齐机器人的 125Hz）
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)



# ============================================
# 启动线程
# ============================================

thread = threading.Thread(
    target=mujoco_loop,
    daemon=True
)

thread.start()

# ============================================
# DearPyGui 主循环
# ============================================

while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

dpg.destroy_context()