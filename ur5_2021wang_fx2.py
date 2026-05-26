'''
论文复现
Wang 等 - 2021 - Simultaneous Calibration of Multicoordinates for a Dual-Robot System by Solving the AXB = YCZ Proble
# TODO: 适配 Ubuntu 22.04 下的 ROS 2 Humble 接口
方向二：视觉重投影误差驱动的全局优化 (Bundle Adjustment)
'''
import os
os.environ['MUJOCO_GL'] = 'glfw' # 确保使用 glfw 后端

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import scipy.linalg
import sys
import datetime

from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

import utils_robot
from utils_vision import VisionModule
from utils_logging import DoubleLogger

# ================== 核心算法数学模块 ==================

def vec(mat):
    """
    Eq. (8): vec() 算子，按列展开矩阵为向量 (Fortran order)
    """
    return mat.flatten(order='F').reshape(-1, 1)

def unvec(v, shape=(3, 3)):
    """
    vec() 的逆操作，将向量恢复为矩阵
    """
    return v.reshape(shape, order='F')

def project_to_SO3(M):
    """
    流行投影|利用 SVD 将不严格正交的 3x3 矩阵投影到标准 SO(3) 旋转矩阵上
    保证行列式为 +1
    """
    U, _, Vh = np.linalg.svd(M)
    R_ortho = U @ Vh
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vh
    return R_ortho

def solve_rotation_core(M_tilde_input):
    """
    封装之前的 SVD 和 project_to_SO3 逻辑
    """
    # Eq. (24) - Eq. (26): 对 M_tilde_ABC 进行 SVD 分解，寻找最小特征值对应的特征向量
    # 为了计算效率，通常对 M^T * M 做特征值分解或者对 M 做 SVD
    _, _, Vh = np.linalg.svd(M_tilde_input, full_matrices=False)

    # v_min 是 Vh 的最后一行 (对应最小奇异值的右奇异向量)
    v_min = Vh[-1, :]

    # Eq. (26): m_XYZ = 2 * sqrt(3) * v_min
    m_raw = 2 * np.sqrt(3) * v_min[:9] # 提取前9位

    R_raw = unvec(m_raw) 
  
    # 💡 核心修复：必须在这里修复符号歧义，再扔去正交化投影！
    if np.linalg.det(R_raw) < 0:
        R_raw = -R_raw

    return project_to_SO3(R_raw)

def build_M_and_solve(A_list, B_list, C_list):
    """
    通用构造器：根据给定的 A, B, C 列表，构造 9n x 90 的大矩阵并求解旋转矩阵
    """
    n = len(A_list)
    M_tilde = np.zeros((9 * n, 90))
    
    for i in range(n):
        RA, RB, RC = A_list[i], B_list[i], C_list[i]
        
        # Eq. (11): M_AB = RB^T \otimes RA
        M_AB = np.kron(RB.T, RA)
        
        # 💡 核心修复：修正论文公式，用 Kronecker 积代替循环构造
        M_C = np.kron(np.eye(9), vec(RC).T)
        
        # 横向拼接并填入大矩阵
        M_tilde[i*9 : (i+1)*9, :] = np.hstack([M_AB, -M_C])
        
    # 等所有数据填满后，在循环外部调用求解！
    return solve_rotation_core(M_tilde)

def solve_axb_ycz_closed_form(samples):
    """
    复现论文 Section II-B: Closed Form Method
    基于 Kronecker 积的闭式解
    """
    n = len(samples)
    if n < 3:
        raise ValueError("根据论文 Section II-D, 至少需要3组非共线姿态！")

    # 数据解析
    R_A = [s['A'][:3, :3] for s in samples]
    t_A = [s['A'][:3, 3].reshape(3,1) for s in samples]
    R_B = [s['B'][:3, :3] for s in samples]
    t_B = [s['B'][:3, 3].reshape(3,1) for s in samples]
    R_C = [s['C'][:3, :3] for s in samples]
    t_C = [s['C'][:3, 3].reshape(3,1) for s in samples]

    # ==========================================
    # 求解旋转分量 RX (Eq. 7 - Eq. 27)
    # ==========================================

    # 求解 RX (原始 A, B, C)
    RX = build_M_and_solve(R_A, R_B, R_C)
    
    # 求解 RY (将 A 换成 A^T, B 换成 C, C 换成 B)
    RA_T = [r.T for r in R_A]
    RY = build_M_and_solve(RA_T, R_C, R_B)
    
    # 求解 RZ (将 A 换成 C, B 换成 B^T, C 换成 A)
    RB_T = [r.T for r in R_B]
    RZ = build_M_and_solve(R_C, RB_T, R_A)
    
    # ==========================================
    # 求解平移分量 tX, tY, tZ (Eq. 29 - Eq. 31)
    # ==========================================
    J_tilde = np.zeros((3 * n, 9))
    b_tilde = np.zeros((3 * n, 1))
    
    for i in range(n):
        # Eq. (29): J = [RA, -I, -RY*RC]
        J_i = np.hstack([R_A[i], -np.eye(3), -RY @ R_C[i]])
        # Eq. (29): b = RY*tC - tA - RA*RX*tB
        b_i = RY @ t_C[i] - t_A[i] - R_A[i] @ RX @ t_B[i]
        
        J_tilde[i*3 : (i+1)*3, :] = J_i
        b_tilde[i*3 : (i+1)*3, :] = b_i
        
    # Eq. (31): 线性最小二乘法求解平移向量
    t_XYZ, _, _, _ = np.linalg.lstsq(J_tilde, b_tilde, rcond=None)
    tX, tY, tZ = t_XYZ[0:3], t_XYZ[3:6], t_XYZ[6:9]
    
    # 组合齐次矩阵
    X, Y, Z = np.eye(4), np.eye(4), np.eye(4)
    X[:3,:3], X[:3,3] = RX, tX.flatten()
    Y[:3,:3], Y[:3,3] = RY, tY.flatten()
    Z[:3,:3], Z[:3,3] = RZ, tZ.flatten()

    print("RX det:", np.linalg.det(RX))
    print("RY det:", np.linalg.det(RY))
    print("RZ det:", np.linalg.det(RZ))
    
    return X, Y, Z


def solve_axb_ycz_iterative(samples, X0, Y0, Z0):
    """ 
    复现论文 Section II-C: Iterative Method 的核心思想
    这里舍弃了繁琐的 SVRG，使用工业界标定常用的 李代数 + LM 算法
    """
    print("开始全局非线性迭代优化 (Lie Algebra + LM)...")
    
    # 辅助函数：4x4 矩阵 <-> 6D 向量 [rx, ry, rz, tx, ty, tz]
    def mat2vec(M):
        r = R.from_matrix(M[:3, :3]).as_rotvec()
        return np.hstack((r, M[:3, 3]))
        
    def vec2mat(v):
        M = np.eye(4)
        M[:3, :3] = R.from_rotvec(v[:3]).as_matrix()
        M[:3, 3] = v[3:6]
        return M

    # 初始参数 (18 维向量)
    p0 = np.hstack((mat2vec(X0), mat2vec(Y0), mat2vec(Z0)))

    # 定义残差函数 (最小化 || AXB - YCZ ||_F)
    def residual(p, samples):
        X = vec2mat(p[0:6])
        Y = vec2mat(p[6:12])
        Z = vec2mat(p[12:18])
        
        errs = []

        for s in samples:
            A, B, C = s['A'], s['B'], s['C']
            # 仅取前 3 行 (忽略齐次项末尾的 [0,0,0,1])
            diff = (A @ X @ B - Y @ C @ Z)[:3, :]
            errs.extend(diff.flatten())
        return np.array(errs)

    # 运行非线性最小二乘优化
    res = scipy.optimize.least_squares(residual, p0, args=(samples,), method='lm', max_nfev=2000)
    
    X_opt = vec2mat(res.x[0:6])
    Y_opt = vec2mat(res.x[6:12])
    Z_opt = vec2mat(res.x[12:18])
    
    return X_opt, Y_opt, Z_opt


def solve_axb_ycz_bundle_adjustment(samples, X0, Y0, Z0, K, dist_coeffs):
    """
    【创新点核心】基于 2D 像素重投影误差的全局光束法平差 (Bundle Adjustment)
    跳过 PnP 的中间矩阵 B，实现端到端的物理-像素联合标定！
    """
    print("\n🚀 开始执行基于像素重投影误差的 BA 全局优化...")
    
    def mat2vec(M):
        r = R.from_matrix(M[:3, :3]).as_rotvec()
        return np.hstack((r, M[:3, 3]))
        
    def vec2mat(v):
        M = np.eye(4)
        M[:3, :3] = R.from_rotvec(v[:3]).as_matrix()
        M[:3, 3] = v[3:6]
        return M

    # 初始参数 (18 维向量，由王等人的闭式解提供高质量初始值)
    p0 = np.hstack((mat2vec(X0), mat2vec(Y0), mat2vec(Z0)))

    # 定义残差函数：计算所有角点的 2D 像素误差
    def reprojection_residual(p, samples):
        X = vec2mat(p[0:6])
        Y = vec2mat(p[6:12])
        Z = vec2mat(p[12:18])
        
        errs = []
        for s in samples:
            A, C = s['A'], s['C']
            img_points_obs = s['img_points'] # 真实的 2D 像素点
            obj_points = s['obj_points']     # 标定板的 3D 物理点
            
            # 核心数学推导：预测相机到标定板的变换矩阵 B_pred
            # B_pred = (A * X)^-1 * (Y * C * Z)
            AX = A @ X
            YCZ = Y @ C @ Z
            B_pred = np.linalg.inv(AX) @ YCZ
            
            # 提取旋转和平移向量，准备投影
            rvec_pred = R.from_matrix(B_pred[:3, :3]).as_rotvec()
            tvec_pred = B_pred[:3, 3]
            
            # 使用内参 K 将 3D 点投影到像素平面
            img_points_pred, _ = cv2.projectPoints(
                obj_points, rvec_pred, tvec_pred, K, dist_coeffs
            )
            
            # 计算像素偏差 (U, V 方向的误差差值)
            # diff 是一个 1D 数组，包含了该帧图像上所有角点的像素误差
            diff = (img_points_obs.flatten() - img_points_pred.flatten())
            errs.extend(diff)
            
        return np.array(errs)

    # 运行强鲁棒性的非线性优化
    # 由于是像素误差（没有前面那个极大的 overflow 问题），这里可以直接放心使用 trf + huber！
    res = scipy.optimize.least_squares(
        reprojection_residual, 
        p0, 
        args=(samples,), 
        method='trf', 
        loss='huber',  # 抗噪核函数，彻底屏蔽残次照片！
        f_scale=1.0, 
        max_nfev=2000
    )
    
    X_ba = vec2mat(res.x[0:6])
    Y_ba = vec2mat(res.x[6:12])
    Z_ba = vec2mat(res.x[12:18])
    
    # 顺便统计一下优化后的平均像素重投影误差
    final_residuals = reprojection_residual(res.x, samples)
    mean_pixel_err = np.mean(np.abs(final_residuals))
    print(f"🎯 BA 优化完成！平均重投影误差: {mean_pixel_err:.4f} 像素")
    
    return X_ba, Y_ba, Z_ba



# ================== 仿真与数据采集模块 ==================

def add_noise_to_matrix(T, rot_noise_deg=0.1, trans_noise_mm=0.5):
    """模拟真实机械臂的读取误差"""
    T_noisy = T.copy()
    
    # 注入平移噪声 (例如 +/- 0.5 mm)
    noise_t = np.random.uniform(-trans_noise_mm/1000.0, trans_noise_mm/1000.0, 3)
    T_noisy[:3, 3] += noise_t
    
    # 注入旋转噪声 (例如 +/- 0.1 度)
    noise_r_vec = np.random.uniform(-np.radians(rot_noise_deg), np.radians(rot_noise_deg), 3)
    R_noise = R.from_rotvec(noise_r_vec).as_matrix()
    T_noisy[:3, :3] = R_noise @ T_noisy[:3, :3]
    
    return T_noisy

def get_site_4x4_mat(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mat = np.eye(4)
    mat[:3, :3] = data.site(site_id).xmat.reshape(3, 3)
    mat[:3, 3] = data.site(site_id).xpos
    return mat

def get_body_4x4_mat(model, data, body_name):
    """专门用来获取 body (连杆/基座) 的 4x4 矩阵"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"❌ 找不到名为 '{body_name}' 的 body，请检查 XML 文件！")
        
    mat = np.eye(4)
    mat[:3, :3] = data.body(body_id).xmat.reshape(3, 3)
    mat[:3, 3] = data.body(body_id).xpos
    return mat

def print_current_status(model, data, q1_idx, q2_idx):
    q1 = data.qpos[q1_idx]
    q2 = data.qpos[q2_idx]
    
    tcp1_mat = get_site_4x4_mat(model, data, "tcp")
    tcp2_mat = get_site_4x4_mat(model, data, "tcp_2")
    
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

def verify_calibration_results(samples, X, Y, Z, model, data):
    """
    验证标定结果的准确性
    """
    print("\n" + "="*50)
    print("📊 标定结果精度验证 (Validation)")
    print("="*50)

    # ---------------------------------------------------------
    # 1. 计算方程闭环残差 ||AXB - YCZ|| (论文常用指标)
    # ---------------------------------------------------------
    trans_errors = []
    rot_errors = []
    
    for s in samples:
        A, B, C = s['A'], s['B'], s['C']
        
        # 左侧: 从 Robot1 算出的标定板在世界坐标系下的位姿
        Left = A @ X @ B
        # 右侧: 从 Robot2 算出的标定板在世界坐标系下的位姿
        Right = Y @ C @ Z
        
        # 计算平移偏差 (距离)
        t_err = np.linalg.norm(Left[:3, 3] - Right[:3, 3])
        trans_errors.append(t_err)
        
        # 计算旋转偏差 (夹角)
        R_err = Left[:3, :3] @ Right[:3, :3].T
        angle_rad = np.linalg.norm(R.from_matrix(R_err).as_rotvec())
        rot_errors.append(np.rad2deg(angle_rad))
        
    mean_t_err = np.mean(trans_errors) * 1000 # 转为毫米
    mean_r_err = np.mean(rot_errors)
    max_t_err = np.max(trans_errors) * 1000
    max_r_err = np.max(rot_errors)
    
    print("\n[1] 数学闭环残差 (AXB vs YCZ):")
    print(f"    ▶ 平均平移误差: {mean_t_err:.3f} mm  (最大: {max_t_err:.3f} mm)")
    print(f"    ▶ 平均旋转误差: {mean_r_err:.4f} °   (最大: {max_r_err:.4f} °)")

    # ---------------------------------------------------------
    # 2. 对比 MuJoCo 上帝视角的真值 (Ground Truth)
    # ---------------------------------------------------------
    print("\n[2] 与 MuJoCo 真实物理真值对比:")
    
    # 【获取 X 的真值 (TCP1 -> Camera)】
    try:
        # 💡 核心修复 3: 使用正确的 API 获取 camera 对象的真实位姿
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
        T_cam_world = np.eye(4)
        T_cam_world[:3, :3] = data.cam(cam_id).xmat.reshape(3, 3)
        T_cam_world[:3, 3] = data.cam(cam_id).xpos
        
        site_tcp1 = "tcp"
        T_tcp1_world = get_site_4x4_mat(model, data, site_tcp1)

        # X_gt = (TCP_world)^-1 * Cam_world
        X_gt = np.linalg.inv(T_tcp1_world) @ T_cam_world
        print("    ▶ MuJoCo 上帝视角真值 X_gt (TCP1 -> Camera):\n", np.round(X_gt, 8))
        
        X_t_err = np.linalg.norm(X[:3, 3] - X_gt[:3, 3]) * 1000
        R_err_X = X[:3, :3] @ X_gt[:3, :3].T
        X_r_err = np.rad2deg(np.linalg.norm(R.from_matrix(R_err_X).as_rotvec()))
        
        print(f"    ▶ 手眼矩阵 X 平移误差: {X_t_err:.3f} mm, 旋转误差: {X_r_err:.4f} °")
        
    except Exception as e:
        print("    (无法获取 X 的真值，请检查 site 名称是否对应)")


    # 【获取 Y 的真值 (Base1 -> Base2)】
    try:
        T_world_base1 = get_body_4x4_mat(model, data, "base")
        T_world_base2 = get_body_4x4_mat(model, data, "base_2")
        
        # Y_gt = (World -> Base1)^-1 * (World -> Base2) = Base1 -> Base2
        Y_gt = np.linalg.inv(T_world_base1) @ T_world_base2
        
        Y_t_err = np.linalg.norm(Y[:3, 3] - Y_gt[:3, 3]) * 1000
        R_err_Y = Y[:3, :3] @ Y_gt[:3, :3].T
        Y_r_err = np.rad2deg(np.linalg.norm(R.from_matrix(R_err_Y).as_rotvec()))
        print(f"    ▶ 基座矩阵 Y 平移误差: {Y_t_err:.3f} mm, 旋转误差: {Y_r_err:.4f} °")
        
    except Exception as e:
        print("    (无法获取 Y 的真值，请检查基座 site 名称是否对应)")    




# ===================================== 参数设置 =====================================
XML_PATH = "/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/scene_dualcalib.xml"
N_SAMPLES = 100 

HOME_QPOS_1 = np.array([-1.63, -2.07, 2.01, -3.14, -1.57, 0])
HOME_QPOS_2 = np.array([1.19, -1.51, 1.51, -3.14, -1.57, 0])

def main():
 
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    width, height = 640, 480
    vision = VisionModule(width, height, fovy=50) 
    renderer = mujoco.Renderer(model, height, width)
    
    # 图像记录
    samples = []
    start_collecting = False 

    q1_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"shoulder_pan_joint") + i for i in range(6)]
    q2_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"shoulder_pan_joint_2") + i for i in range(6)]

    # 启动时的初始关节赋值
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
    print("▶ 调好姿势后，选中弹出的 Camera View 窗口按下快捷键:")
    print("   [c] - 锁定当前姿态，关闭物理干扰，开始纯运动学自动扰动采集！")
    print("   [p] - 打印当前 TCP 位姿和关节角度")
    print("   [q] - 提前结束采集 / 退出程序")
    print("===============================================================")

    # 检测并纠正180 度翻转的二义性
    flip_count = 0
    # 在进入循环前，定义一个锚点旋转矩阵
    anchor_R_cv = None

    with DoubleLogger(log_dir="calibration_logs") as logger:
        with mujoco.viewer.launch_passive(model, data) as viewer:

            while len(samples) < N_SAMPLES and viewer.is_running():
                
                if not start_collecting:
                    # ====== 正常物理运行状态 ======
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

                    # ====== 自动扰动采集 ======
                    # 此时不再调用 mj_step()，完全剥离物理法则（没有重力，无视碰撞）
                    # 直接“瞬移”关节进行纯运动学的纯净采样，避免物理回弹报错
                    q1 = base_qpos_1 + np.random.uniform(-0.6, 0.6, 6)
                    q2 = base_qpos_2 + np.random.uniform(-0.6, 0.6, 6)
                    
                    data.qpos[q1_idx] = q1
                    data.qpos[q2_idx] = q2
                    mujoco.mj_forward(model, data)  # 仅更新运动学正解和相机位置
                    
                    renderer.update_scene(data, "wrist_camera")
                    img_rgb = renderer.render()
                    found, rvec, tvec, img_draw = vision.detect_and_pose(img_rgb)
                    
                    if found:

                        # 构造保存路径（建议在 main 开头定义好 IMAGE_SAVE_DIR）
                        # 假设你已经创建了文件夹：os.makedirs("calibration_logs", exist_ok=True)
                        sample_idx = len(samples) + 1
                        save_path = f"calibration_logs/samples/sample_{sample_idx:03d}.png"
                        
                        # 获取带有角点和坐标轴标注的图像
                        # 注意：你的 detect_and_pose 返回的 img_bgr 已经是 BGR 格式了
                        # 直接使用 cv2.imwrite 即可，不需要再转换颜色空间
                        cv2.imwrite(save_path, img_draw)

                        R_cv, _ = cv2.Rodrigues(rvec)
                        t_cv = tvec.flatten()

                         # =======================================================
                        # 💡 纯 OpenCV 坐标系下的 180 度翻转诊断与修复
                        # =======================================================
                        if anchor_R_cv is None:
                            # 记录第一帧的旋转作为锚点
                            anchor_R_cv = R_cv.copy()
                        else:
                            # 提取锚点和当前的 X 轴分量 (OpenCV 下第一列是 X 轴)
                            x_axis_anchor = anchor_R_cv[:, 0]
                            x_axis_current = R_cv[:, 0]
                            
                            # 如果夹角超过 90 度，说明发生了 180 度翻转
                            if np.dot(x_axis_anchor, x_axis_current) < 0:
                                flip_count += 1
                                
                                # OpenCV 坐标系下的翻转矩阵 (绕 Z 轴转 180 度)
                                R_flip_180_cv = np.array([
                                    [-1,  0,  0],
                                    [ 0, -1,  0],
                                    [ 0,  0,  1]
                                ])

                                # 计算物理偏移向量 (标定板的长和宽)
                                offset_cv = np.array([
                                    (vision.pattern_size[0] - 1) * vision.square_size,
                                    (vision.pattern_size[1] - 1) * vision.square_size,
                                    0.0
                                ])

                                # 💡 核心修复：必须使用【未翻转前的 R_cv】来计算平移！
                                # 把原点从错误的“右下角”挪回“左上角”
                                t_cv_corrected = t_cv + R_cv @ offset_cv
                                # 然后再翻转坐标轴的方向
                                R_cv_corrected = R_cv @ R_flip_180_cv

                                # 将修复后的结果转回向量格式，准备画图
                                rvec_corrected, _ = cv2.Rodrigues(R_cv_corrected)
                                tvec_corrected = t_cv_corrected.reshape(3, 1)

                                # 画图验证 (拿干净的 RGB 转 BGR)
                                img_corrected_draw = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                                
                                # ⚠️ 核心修复：接住 drawFrameAxes 的返回值 img_corrected_draw
                                img_corrected_draw = cv2.drawFrameAxes(
                                    img_corrected_draw, 
                                    vision.K, 
                                    vision.dist_coeffs, 
                                    rvec_corrected, 
                                    tvec_corrected, 
                                    0.1,  # 坐标轴长度 0.1 米
                                    3     # 线条粗细
                                )

                                cv2.putText(img_corrected_draw, "FIXED: 180 Deg Flip Corrected!", 
                                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                                # 保存翻转修复图
                                sample_idx = len(samples) + 1
                                flip_path = os.path.join("calibration_logs", "samples", f"sample_{sample_idx:03d}_r.png")
                                cv2.imwrite(flip_path, img_corrected_draw)

                                # 覆盖原来的变量，让下游的 MuJoCo 计算拿到的都是正确数据
                                R_cv = R_cv_corrected
                                t_cv = t_cv_corrected
                                
                        # ==================================================
                        # OpenCV -> MuJoCo/OpenGL 坐标系翻转 (转给机器人 B 矩阵)
                        # ==================================================
                        T_cv2gl = np.array([
                            [1,  0,  0],
                            [0, -1,  0],
                            [0,  0, -1]
                        ])
                        R_b = T_cv2gl @ R_cv
                        t_b = T_cv2gl @ t_cv

                        B = np.eye(4)
                        B[:3, :3] = R_b
                        B[:3, 3] = t_b

                        T_world_base1 = get_body_4x4_mat(model, data, "base")
                        T_world_base2 = get_body_4x4_mat(model, data, "base_2")

                        T_world_tcp1 = get_site_4x4_mat(model, data, "tcp")
                        T_world_tcp2 = get_site_4x4_mat(model, data, "tcp_2")

                        # 计算 TCP 相对于自己基座的坐标
                        A_real = np.linalg.inv(T_world_base1) @ T_world_tcp1
                        C_real = np.linalg.inv(T_world_base2) @ T_world_tcp2

                        samples.append({
                            'A': A_real,      
                            'B': B,                                    
                            'C': C_real     
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

        # 最后关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()


        # ====== 采集结束，开始解算 ======
        if len(samples) < 5:
            print("\n采集的数据太少，无法计算标定矩阵。")
            return

        print(f"\n✅ 采集结束，共 {N_SAMPLES} 组数据。")
        print(f"🔧 [视觉诊断] 自动触发 180 度翻转纠正次数: {flip_count} 次")
        
        print("\n================ 1. 闭式解 (Closed Form) ================")

        X_cf, Y_cf, Z_cf = solve_axb_ycz_closed_form(samples)
        print("X (手眼矩阵 Robot1 Flange -> Camera):\n", np.round(X_cf, 8))
        print("-" * 40)
        print("Y (基座转换 Robot1 Base -> Robot2 Base):\n", np.round(Y_cf, 8))
        print("-" * 40)
        print("Z (抓取矩阵 Robot2 Flange -> Board):\n", np.round(Z_cf, 8))
        
        print("\n================ 2. 迭代解 (Iterative Optimization) ================")

        X_opt, Y_opt, Z_opt = solve_axb_ycz_iterative(samples, X_cf, Y_cf, Z_cf)
        print("最优 X (手眼矩阵 Robot1 Flange -> Camera):\n", np.round(X_opt, 8))
        print("-" * 40)
        print("最优 Y (基座转换 Robot1 Base -> Robot2 Base):\n", np.round(Y_opt, 8))
        print("-" * 40)
        print("最优 Z (抓取矩阵 Robot2 Flange -> Board):\n", np.round(Z_opt, 8))
        print("================================================================")

        # 验证时传入迭代优化后的 X, Y, Z
        verify_calibration_results(samples, X_opt, Y_opt, Z_opt, model, data)

        # 调用类方法，一键保存脱机数据
        logger.save_calibra_raw_data_offline(samples, X_opt, Y_opt, Z_opt)


if __name__ == "__main__":
    main()