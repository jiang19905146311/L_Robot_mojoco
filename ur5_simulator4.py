'''
双机器人视觉标定
'''
import os
os.environ['MUJOCO_GL'] = 'glfw' # 确保使用 glfw 后端

import mujoco
import mujoco.viewer
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import scipy.linalg

from utils_vision import VisionModule

# ================== 核心算法数学模块 ==================

def pose2vec(T):
    rvec = R.from_matrix(T[:3, :3]).as_rotvec()
    tvec = T[:3, 3]
    return np.concatenate([rvec, tvec])

def vec2pose(vec):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(vec[:3]).as_matrix()
    T[:3, 3] = vec[3:]
    return T

def error_function(params, samples):
    X = vec2pose(params[0:6])
    Y = vec2pose(params[6:12])
    Z = vec2pose(params[12:18])
    
    residuals = []
    for s in samples:
        A, B, C = s['A'], s['B'], s['C']
        AXB = A @ X @ B
        YCZ = Y @ C @ Z
        t_err = AXB[:3, 3] - YCZ[:3, 3]
        R_err_mat = AXB[:3, :3] @ YCZ[:3, :3].T
        r_err = R.from_matrix(R_err_mat).as_rotvec()
        residuals.extend(t_err)
        residuals.extend(r_err)
        
    return np.array(residuals)

def solve_axb_ycz_optimization(samples):
    initial_guess = np.zeros(18) 
    print("\n正在运行 LM 非线性优化算法求解 X, Y, Z...")
    result = least_squares(
        error_function, 
        initial_guess, 
        args=(samples,), 
        method='lm', 
        max_nfev=10000
    )
    
    X_opt = vec2pose(result.x[0:6])
    Y_opt = vec2pose(result.x[6:12])
    Z_opt = vec2pose(result.x[12:18])
    return X_opt, Y_opt, Z_opt

# ================== 仿真与数据采集模块 ==================

def get_4x4_mat(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mat = np.eye(4)
    mat[:3, :3] = data.site(site_id).xmat.reshape(3, 3)
    mat[:3, 3] = data.site(site_id).xpos
    return mat

def print_current_status(model, data, q1_idx, q2_idx):
    q1 = data.qpos[q1_idx]
    q2 = data.qpos[q2_idx]
    
    tcp1_mat = get_4x4_mat(model, data, "tcp")
    tcp2_mat = get_4x4_mat(model, data, "tcp_2")
    
    tcp1_pos = tcp1_mat[:3, 3]
    tcp1_euler = R.from_matrix(tcp1_mat[:3, :3]).as_euler('xyz', degrees=True)
    
    tcp2_pos = tcp2_mat[:3, 3]
    tcp2_euler = R.from_matrix(tcp2_mat[:3, :3]).as_euler('xyz', degrees=True)
    
    print("\n" + "="*45)
    print("📍 [Robot 1 - Camera] 当前状态:")
    print(f"   关节角度 (rad): {np.round(q1, 4)}")
    print(f"   TCP 位置 (m):   X={tcp1_pos[0]:.4f}, Y={tcp1_pos[1]:.4f}, Z={tcp1_pos[2]:.4f}")
    print(f"   TCP 欧拉角 (deg): Rx={tcp1_euler[0]:.2f}, Ry={tcp1_euler[1]:.2f}, Rz={tcp1_euler[2]:.2f}")
    print("-" * 45)
    print("📍 [Robot 2 - Board] 当前状态:")
    print(f"   关节角度 (rad): {np.round(q2, 4)}")
    print(f"   TCP 位置 (m):   X={tcp2_pos[0]:.4f}, Y={tcp2_pos[1]:.4f}, Z={tcp2_pos[2]:.4f}")
    print(f"   TCP 欧拉角 (deg): Rx={tcp2_euler[0]:.2f}, Ry={tcp2_euler[1]:.2f}, Rz={tcp2_euler[2]:.2f}")
    print("="*45 + "\n")


# === 参数设置 ===
XML_PATH = "/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/scene_dualcalib.xml"
N_SAMPLES = 100 

HOME_QPOS_1 = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0])
HOME_QPOS_2 = np.array([1.57, 0, 1.8, -2.0, -1.57, 0])

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    width, height = 640, 480
    vision = VisionModule(width, height, fovy=50) 
    renderer = mujoco.Renderer(model, height, width)

    samples = []
    start_collecting = False 

    q1_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"shoulder_pan_joint") + i for i in range(6)]
    q2_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"shoulder_pan_joint_2") + i for i in range(6)]

    # 1. 启动时的初始关节赋值
    data.qpos[q1_idx] = HOME_QPOS_1
    data.qpos[q2_idx] = HOME_QPOS_2
    
    # 同步控制目标 (防止有位置控制器的模型刚启动时剧烈回弹)
    if model.nu >= 12:
        data.ctrl[0:6] = HOME_QPOS_1
        data.ctrl[6:12] = HOME_QPOS_2

    base_qpos_1 = np.copy(HOME_QPOS_1)
    base_qpos_2 = np.copy(HOME_QPOS_2)

    print("===============================================================")
    print("▶ 仿真已启动！目前处于【正常物理运行状态】。")
    print("▶ 您可以用以下方式调整机器人的姿态，直到视野完美：")
    print("   1. 鼠标悬停在机械臂上，按住 [Ctrl] + [鼠标右键] 拖拽它。")
    print("   2. 在右侧面板中，点击 'Control' 选项卡，拖动滑块来控制。")
    print("---------------------------------------------------------------")
    print("▶ 调好姿势后，选中弹出的 Camera View 窗口按下快捷键:")
    print("   [c] - 锁定当前姿态，关闭物理干扰，开始纯运动学自动扰动采集！")
    print("   [p] - 打印当前 TCP 位姿和关节角度")
    print("   [q] - 提前结束采集 / 退出程序")
    print("===============================================================")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while len(samples) < N_SAMPLES and viewer.is_running():
            
            if not start_collecting:
                # ====== 状态 1：正常物理运行状态 ======
                # 为了配合相机的视觉处理延时（约30ms），每次循环让物理引擎多跑几步
                # 这样保证仿真时间流逝正常，机器人是受物理法则控制的（可以被鼠标拖拽）
                for _ in range(15):
                    mujoco.mj_step(model, data)
                
                renderer.update_scene(data, "wrist_camera")
                img_rgb = renderer.render()
                found, rvec, tvec, img_draw = vision.detect_and_pose(img_rgb)
                
                status_color = (0, 255, 0) if found else (0, 0, 255)
                status_text = "Status: PHYSICS RUNNING (Target Found)" if found else "Status: PHYSICS RUNNING (Target Lost!)"
                cv2.putText(img_draw, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(img_draw, "Drag robot with [Ctrl + Right Click]", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.imshow("Camera View", img_draw)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if not found:
                        print("\n⚠ 警告: 当前未识别到标定板，请继续调整姿态！")
                    else:
                        start_collecting = True
                        # 重点：按下C键的一瞬间，记录当前的实际物理关节位置
                        base_qpos_1 = np.copy(data.qpos[q1_idx])
                        base_qpos_2 = np.copy(data.qpos[q2_idx])
                        print("\n▶ 已锁定当前姿态！物理引擎已剥离，开始纯运动学自动扰动采集数据...")
                        
                elif key == ord('p'):
                    print_current_status(model, data, q1_idx, q2_idx)
                elif key == ord('q'):
                    print("\n▶ 用户手动终止程序。")
                    break

            else:
                # ====== 状态 2：自动扰动采集 ======
                # 重点：此时不再调用 mj_step()，完全剥离物理法则（没有重力，无视碰撞）
                # 我们直接“瞬移”关节进行纯运动学的纯净采样，避免物理回弹报错
                q1 = base_qpos_1 + np.random.uniform(-0.3, 0.3, 6)
                q2 = base_qpos_2 + np.random.uniform(-0.3, 0.3, 6)
                
                data.qpos[q1_idx] = q1
                data.qpos[q2_idx] = q2
                mujoco.mj_forward(model, data)  # 仅更新运动学正解和相机位置
                
                renderer.update_scene(data, "wrist_camera")
                img_rgb = renderer.render()
                found, rvec, tvec, img_draw = vision.detect_and_pose(img_rgb)
                
                if found:
                    R_b, _ = cv2.Rodrigues(rvec)
                    B = np.eye(4)
                    B[:3, :3] = R_b
                    B[:3, 3] = tvec.flatten()
                    
                    samples.append({
                        'A': get_4x4_mat(model, data, "tcp"),      
                        'B': B,                                    
                        'C': get_4x4_mat(model, data, "tcp_2")     
                    })
                    
                    cv2.putText(img_draw, f"Collecting: {len(samples)}/{N_SAMPLES}", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(f"进度: {len(samples)}/{N_SAMPLES}", end='\r')
                else:
                    cv2.putText(img_draw, "Target Lost! Retrying...", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.imshow("Camera View", img_draw)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'):
                    print_current_status(model, data, q1_idx, q2_idx)
                elif key == ord('q'):
                    print("\n▶ 用户手动终止程序。")
                    break
            
            viewer.sync()
            
    cv2.destroyAllWindows()
    
    if len(samples) < 5:
        print("\n采集的数据太少，无法计算标定矩阵。")
        return

    print("\n采集结束，准备解算...")
    X, Y, Z = solve_axb_ycz_optimization(samples)
    
    print("\n================ 标定结果 ================")
    print("X (手眼矩阵 Robot1 Flange -> Camera):\n", np.round(X, 5))
    print("-" * 40)
    print("Y (基座转换 Robot1 Base -> Robot2 Base):\n", np.round(Y, 5))
    print("-" * 40)
    print("Z (抓取矩阵 Robot2 Flange -> Board):\n", np.round(Z, 5))
    print("==========================================")

if __name__ == "__main__":
    main()