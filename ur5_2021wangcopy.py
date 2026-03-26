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

# ================== 阶段一：Kronecker 闭式解 ==================

def vec(mat):
    return mat.flatten(order='F').reshape(-1, 1)

def unvec(v, shape=(3, 3)):
    return v.reshape(shape, order='F')

def project_to_SO3(M):
    U, _, Vh = np.linalg.svd(M)
    R_ortho = U @ Vh
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vh
    return R_ortho

def solve_rotation_core(M_tilde_input):
    _, _, Vh = np.linalg.svd(M_tilde_input, full_matrices=False)
    v_min = Vh[-1, :]
    m_raw = 2 * np.sqrt(3) * v_min[:9] 
    R_raw = unvec(m_raw) 
    if np.linalg.det(R_raw) < 0:
        R_raw = -R_raw
    return project_to_SO3(R_raw)

def build_M_and_solve(A_list, B_list, C_list):
    n = len(A_list)
    M_tilde = np.zeros((9 * n, 90))
    for i in range(n):
        M_AB = np.kron(B_list[i].T, A_list[i])
        M_C = np.kron(vec(C_list[i]).T, np.eye(9))
        M_tilde[i*9 : (i+1)*9, :] = np.hstack([M_AB, -M_C])
    return solve_rotation_core(M_tilde)

def solve_axb_ycz_closed_form(samples):
    n = len(samples)
    R_A = [s['A'][:3, :3] for s in samples]
    t_A = [s['A'][:3, 3].reshape(3,1) for s in samples]
    R_B = [s['B_GT'][:3, :3] for s in samples] # 💡 这里强制使用上帝真值 B_GT 求解！
    t_B = [s['B_GT'][:3, 3].reshape(3,1) for s in samples]
    R_C = [s['C'][:3, :3] for s in samples]
    t_C = [s['C'][:3, 3].reshape(3,1) for s in samples]

    RX = build_M_and_solve(R_A, R_B, R_C)
    RY = build_M_and_solve([r.T for r in R_A], R_C, R_B)
    RZ = build_M_and_solve(R_C, [r.T for r in R_B], R_A)
    
    J_tilde = np.zeros((3 * n, 9))
    b_tilde = np.zeros((3 * n, 1))
    for i in range(n):
        J_i = np.hstack([R_A[i], -np.eye(3), -RY @ R_C[i]])
        b_i = RY @ t_C[i] - t_A[i] - R_A[i] @ RX @ t_B[i]
        J_tilde[i*3 : (i+1)*3, :] = J_i
        b_tilde[i*3 : (i+1)*3, :] = b_i
        
    t_XYZ, _, _, _ = np.linalg.lstsq(J_tilde, b_tilde, rcond=None)
    X, Y, Z = np.eye(4), np.eye(4), np.eye(4)
    X[:3,:3], X[:3,3] = RX, t_XYZ[0:3].flatten()
    Y[:3,:3], Y[:3,3] = RY, t_XYZ[3:6].flatten()
    Z[:3,:3], Z[:3,3] = RZ, t_XYZ[6:9].flatten()
    return X, Y, Z

# ================== 阶段二：非线性迭代优化 ==================

def pose2vec(T):
    return np.concatenate([R.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])

def vec2pose(vec):
    T = np.eye(4); T[:3, :3] = R.from_rotvec(vec[:3]).as_matrix(); T[:3, 3] = vec[3:]
    return T

def error_function(params, samples):
    X = vec2pose(params[0:6]); Y = vec2pose(params[6:12]); Z = vec2pose(params[12:18])
    res = []
    for s in samples:
        A, B, C = s['A'], s['B_GT'], s['C'] # 依然使用真值验证
        AXB = A @ X @ B; YCZ = Y @ C @ Z
        res.extend(AXB[:3, 3] - YCZ[:3, 3])
        res.extend(R.from_matrix(AXB[:3, :3] @ YCZ[:3, :3].T).as_rotvec())
    return np.array(res)

def solve_axb_ycz_iterative(samples, X0, Y0, Z0):
    initial_guess = np.concatenate([pose2vec(X0), pose2vec(Y0), pose2vec(Z0)])
    print("\n▶ 开始第二阶段：非线性迭代优化 (Iterative Optimization)...")
    res = least_squares(error_function, initial_guess, args=(samples,), method='lm', max_nfev=5000)
    return vec2pose(res.x[0:6]), vec2pose(res.x[6:12]), vec2pose(res.x[12:18])

# ================== 仿真、采集与验证模块 ==================

def get_4x4_mat(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mat = np.eye(4); mat[:3, :3] = data.site(site_id).xmat.reshape(3, 3); mat[:3, 3] = data.site(site_id).xpos
    return mat

def get_gt_B_matrix(model, data):
    """ 提取 MuJoCo 物理底层的上帝真值 B 矩阵 """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "checkerboard_attached")
    T_board = np.eye(4); T_board[:3, :3] = data.xmat[body_id].reshape(3, 3); T_board[:3, 3] = data.xpos[body_id]
    
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    T_cam = np.eye(4); T_cam[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3); T_cam[:3, 3] = data.cam_xpos[cam_id]
    
    return np.linalg.inv(T_cam) @ T_board

def verify_results(samples, X, Y, Z, model, data, phase_name=""):
    print(f"\n================ {phase_name} 精度验证 ================")
    t_errs, r_errs = [], []
    for s in samples:
        Left = s['A'] @ X @ s['B_GT']
        Right = Y @ s['C'] @ Z
        t_errs.append(np.linalg.norm(Left[:3, 3] - Right[:3, 3]))
        r_errs.append(np.rad2deg(np.linalg.norm(R.from_matrix(Left[:3, :3] @ Right[:3, :3].T).as_rotvec())))
        
    print(f"[1] 方程残差 (AXB vs YCZ): 平均平移 {np.mean(t_errs)*1000:.4f} mm, 平均旋转 {np.mean(r_errs):.4f} °")

    T_tcp1 = get_4x4_mat(model, data, "tcp")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    T_cam_world = np.eye(4); T_cam_world[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3); T_cam_world[:3, 3] = data.cam_xpos[cam_id]
    
    X_gt = np.linalg.inv(T_tcp1) @ T_cam_world
    X_t_err = np.linalg.norm(X[:3, 3] - X_gt[:3, 3]) * 1000
    X_r_err = np.rad2deg(np.linalg.norm(R.from_matrix(X[:3, :3] @ X_gt[:3, :3].T).as_rotvec()))
    
    Y_gt = np.eye(4)
    Y_t_err = np.linalg.norm(Y[:3, 3] - Y_gt[:3, 3]) * 1000
    Y_r_err = np.rad2deg(np.linalg.norm(R.from_matrix(Y[:3, :3] @ Y_gt[:3, :3].T).as_rotvec()))
    
    print(f"[2] 物理真值对比:")
    print(f"    ▶ X 矩阵: 误差 {X_t_err:.4f} mm, {X_r_err:.4f} °")
    print(f"    ▶ Y 矩阵: 误差 {Y_t_err:.4f} mm, {Y_r_err:.4f} °")

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

    samples = []
    start_collecting = False 

    q1_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"shoulder_pan_joint") + i for i in range(6)]
    q2_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"shoulder_pan_joint_2") + i for i in range(6)]

    data.qpos[q1_idx] = HOME_QPOS_1; data.qpos[q2_idx] = HOME_QPOS_2
    if model.nu >= 12: data.ctrl[0:6] = HOME_QPOS_1; data.ctrl[6:12] = HOME_QPOS_2

    base_qpos_1, base_qpos_2 = np.copy(HOME_QPOS_1), np.copy(HOME_QPOS_2)

    print("▶ 物理引擎已启动。按 [c] 键开始无噪声真值数据采集...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while len(samples) < N_SAMPLES and viewer.is_running():
            if not start_collecting:
                for _ in range(15): mujoco.mj_step(model, data)
                renderer.update_scene(data, "wrist_camera"); img_rgb = renderer.render()
                found, rvec, tvec, img_draw = vision.detect_and_pose(img_rgb)
                cv2.imshow("Camera View", img_draw)
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    start_collecting = True
                    base_qpos_1, base_qpos_2 = np.copy(data.qpos[q1_idx]), np.copy(data.qpos[q2_idx])
            else:
                # 扰动设定为比较大的范围，保证矩阵满秩！
                data.qpos[q1_idx] = base_qpos_1 + np.random.uniform(-0.5, 0.5, 6)
                data.qpos[q2_idx] = base_qpos_2 + np.random.uniform(-0.5, 0.5, 6)
                mujoco.mj_forward(model, data)  
                
                renderer.update_scene(data, "wrist_camera"); img_rgb = renderer.render()
                found, rvec, tvec, img_draw = vision.detect_and_pose(img_rgb)
                
                # 无论 OpenCV 有没有跳变，我们直接提取 MuJoCo 底层的绝对真值！
                B_GT = get_gt_B_matrix(model, data)
                
                samples.append({
                    'A': get_4x4_mat(model, data, "tcp"),      
                    'B_GT': B_GT,                                    
                    'C': get_4x4_mat(model, data, "tcp_2")     
                })
                print(f"进度: {len(samples)}/{N_SAMPLES}", end='\r')
                
                cv2.imshow("Camera View", img_draw)
                cv2.waitKey(1)
            viewer.sync()
            
    cv2.destroyAllWindows()
    
    if len(samples) >= 3:
        # 1. 跑闭式解
        X_closed, Y_closed, Z_closed = solve_axb_ycz_closed_form(samples)
        verify_results(samples, X_closed, Y_closed, Z_closed, model, data, "阶段一：闭式解 (Closed-Form)")
        
        # 2. 跑迭代解
        X_opt, Y_opt, Z_opt = solve_axb_ycz_iterative(samples, X_closed, Y_closed, Z_closed)
        verify_results(samples, X_opt, Y_opt, Z_opt, model, data, "阶段二：迭代优化 (Iterative)")

if __name__ == "__main__":
    main()