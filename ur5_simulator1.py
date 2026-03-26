'''
ur5_simulation.ur5_simulator1 的 Docstring
单机械臂抓取仿真+拍照功能
'''
import os
import mujoco
import mujoco.viewer
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# === 配置参数 ===
XML_PATH = "/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/scene_catch.xml"
IK_DAMPING = 1e-2     # IK 阻尼系数
STEP_SIZE = 0.5       # IK 更新步长

GRIPPER_OPEN = 0   # 夹爪张开值
GRIPPER_CLOSE = 0.04   # 夹爪闭合值
Z_HOVER = 0.15      # 悬停高度
Z_GRASP = 0.025    # 抓取高度 (需要根据物体高度微调)

# 全局变量缓存机器人关节的索引，避免每步IK都查表
ROBOT_JOINT_INDICES = None 

def load_ur5():
    """加载UR5机器人"""
    global XML_PATH
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
        print("UR5模型加载成功！")
        return model, data
    
    except Exception as e:
        print(f"Error: 加载模型失败 - {e}")
        return None, None

def init_scene(model, data):
    
    """初始化机器人姿态和物体位置"""
    mujoco.mj_resetData(model, data)

    # === 设置初始姿态 ===
    # 对应关节: [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
    home_qpos = [-1.57, -1.57, 1.57, -1.57, -1.57, 0]
    # 将关节位置写入仿真状态
    data.qpos[:6] = home_qpos
    # 同步设置电机控制指令(ctrl)
    # 如果只设了位置 qpos 但不设 ctrl，ctrl 默认为 0， 仿真一开始电机就会拼命把机器人往 0 度拉，导致炸机。
    data.ctrl[:6] = home_qpos

    # 确保夹爪是张开的
    data.ctrl[6] = GRIPPER_OPEN
    data.ctrl[7] = GRIPPER_OPEN

    print("机器人初始姿态已设置。")

    # 设置方块位置 
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    
    if obj_id != -1:
        # 获取该关节在 qpos 数组中的起始地址
        qpos_adr = model.jnt_qposadr[obj_id]
        # --- 设置位置 (x, y, z) ---
        data.qpos[qpos_adr:qpos_adr+3] = [0.5, 0.0, 0.03]   
        # --- 设置姿态 (w, x, y, z) ---
        # 默认四元数 [1, 0, 0, 0] 表示不旋转
        data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]   

        print(f"小方块位置已设置为: {data.qpos[qpos_adr:qpos_adr+3]}")
    else:
        print("警告：未找到 'object_free' 关节，小方块可能位置不正确。")

    # === 执行一次正向动力学计算 ===
    # 根据新的 qpos 计算所有刚体的空间位置(xpos)
    mujoco.mj_forward(model, data)
    print(">>> 场景初始化完成")

def get_site_pose(model, data, site_name="tcp"):
    """获取末端执行器的位置和旋转矩阵"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos = data.site(site_id).xpos
    mat = data.site(site_id).xmat.reshape(3, 3)
    return pos, mat

def get_robot_joint_indices(model):
    """根据名字动态获取UR5的6个关节索引"""
    # 注意：这里的名字必须和XML里的joint name一致！
    names = [f"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    
    indices = []
    for name in names:
        try:
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id == -1:
                print(f"⚠️ Warning: 没找到关节 {name}, IK可能会出错")
            
            # 获取该关节在 qpos (nv) 系统中的自由度地址
            # 对于单自由度关节，dof_adr 就是矩阵列索引
            dof_adr = model.jnt_dofadr[j_id]
            indices.append(dof_adr)
        except Exception as e:
            print(e)
    return np.array(indices)


def solve_6d_ik(model, data, target_pos, target_quat):
    """
    6自由度逆运动学 (位置 + 姿态)
    target_pos: [x, y, z]
    target_quat: [w, x, y, z] (目标姿态四元数)
    返回:
        delta_q: 6个关节需要的角度增量
    """
    global ROBOT_JOINT_INDICES

    # 第一次运行缓存索引
    if ROBOT_JOINT_INDICES is None:
        ROBOT_JOINT_INDICES = get_robot_joint_indices(model)

    # 获取末端执行器 (attachment_site) 的当前位置
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    
    if site_id == -1:
       print("Error: 未找到site，请检查 XML")
       return np.zeros(6)
    
    # 位置误差
    current_pos = data.site(site_id).xpos
    err_pos = target_pos - current_pos

    # 姿态误差 (将四元数差异转换为旋转向量)
    current_mat = data.site(site_id).xmat.reshape(3, 3)
    current_quat = np.zeros(4)
    mujoco.mju_mat2Quat(current_quat, current_mat.flatten())

    # 计算误差四元数: q_err = q_target * q_current_inverse
    neg_current_quat = current_quat.copy()
    neg_current_quat[1:] *= -1 # 共轭即逆
    err_quat = np.zeros(4)
    mujoco.mju_mulQuat(err_quat, target_quat, neg_current_quat)

    # 将误差四元数转为旋转向量 (vel rotation)
    err_rot = np.zeros(3)
    mujoco.mju_quat2Vel(err_rot, err_quat, 1) # 1 表示时间步长，这里仅作为缩放
    
    # 合并误差向量 (6维)
    error = np.hstack([err_pos, err_rot])
    
    # 雅可比矩阵 (6xN)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    
    # 拼接 位置J 和 旋转J
    J_full = np.vstack([jacp, jacr])
    J_ur5 = J_full[:, ROBOT_JOINT_INDICES] # 只取 UR5 关节对应的列
    
    # 求解阻尼最小二乘
    H = J_ur5.T @ J_ur5 + IK_DAMPING * np.eye(6)
    delta_q = np.linalg.solve(H, J_ur5.T @ error)
    
    return delta_q


# 相机截图功能函数
def capture_and_show(renderer, data):
    print("正在捕获相机图像...")
    
    # 抓取腕部相机 (Eye-in-Hand)
    renderer.update_scene(data, camera="wrist_camera")
    wrist_img = renderer.render()
    
    # 抓取顶部相机 (Global)
    try:
        renderer.update_scene(data, camera="top_camera")
        top_img = renderer.render()
    except Exception:
        print("未找到 top_camera，使用黑色背景代替")
        top_img = np.zeros_like(wrist_img)

    # 使用 Matplotlib 显示
    # 这会弹出一个窗口，不仅能看到，还能保存图片
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(wrist_img)
    axes[0].set_title("Wrist Camera (Robot View)")
    axes[0].axis('off') # 隐藏坐标轴
    
    axes[1].imshow(top_img)
    axes[1].set_title("Top Camera (God View)")
    axes[1].axis('off')
    
    print("图像已显示，关闭弹出的图片窗口以继续仿真...")
    plt.show() # 程序会暂停在这里，直到你关闭窗口

# === 主循环 ===
def run_simulation(model, data):
    """可视化与控制循环 (支持按键触发)"""

    # 目标姿态：垂直向下 (四元数 [w, x, y, z])
    # 对于UR5的标准安装，[0, 1, 0, 0] 通常是末端垂直向下
    target_quat_down = np.array([0, 1, 0, 0])

    renderer = mujoco.Renderer(model, height=480, width=640)

    # 定义控制状态
    state = {
        "should_capture": False,  # 是否需要拍照的标记
        "ik_enabled": False,  # 默认不开启 IK 自动追踪
        "info_text": "当前模式:",

        # --- 状态机变量 ---
        "phase": "APPROACH",     # 当前阶段: APPROACH, DESCEND, GRASP, LIFT
        "phase_start_time": 0.0, # 用于延时控制(如等待夹爪闭合)
        "lift_target_pos": None,  # 用于记录抬起时的固定坐标
    }

    # 定义按键回调函数
    # keycode 是按键的整数代码
    def key_callback(keycode):
        # 32 是空格键的 ASCII 码
        # 49 是1键的 ASCII 码
        if keycode == 32:  
            print("触发拍照")
            state["should_capture"] = True
        
        elif keycode == 49:
            state["ik_enabled"] = not state["ik_enabled"] # 取反，实现切换
            
            if state["ik_enabled"]:
                print(">> 切换模式：已开启 [自动]")
                state["info_text"] = "模式: [自动]"
                # 重置状态机，让它从头开始
                state["phase"] = "APPROACH"
                state["lift_target_pos"] = None
            else:
                print(">> 切换模式：已回到 [手动]")
                state["info_text"] = "模式: [手动]"
                cv2.destroyAllWindows() # 关闭视觉窗口
                # 切换回手动时，保持当前的关节位置，防止跳变
                data.ctrl[:6] = data.qpos[:6]

        # 可以添加其他按键，比如 'P' 键暂停 (P的ASCII是80)
        # elif keycode == 80: 
        #     state["paused"] = not state["paused"]


    # 启动查看器，传入 key_callback
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 设置初始视角
        viewer.cam.distance = 2.0
        viewer.cam.lookat = [0.5, 0, 0.5]
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -20
        
        print("="*50)
        print("仿真已启动！")
        print("🕹️  操作说明：")
        print("   在右侧菜单 'Control' 栏拖动滑条控制机械臂")
        print("   [1] : 📸 拍照并显示")
        print("   [2] : 🔄 切换 手动/自动 模式")
        print("   [Esc] : ❌ 退出")
        print("="*50)
        
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # === 拍照逻辑 ===
            if state["should_capture"]:

                capture_and_show(renderer, data)
               
                # 如果你想按一次键拍一张，保持这行
                # 如果想按一次键开启“实时监控”，可以把这行注释掉
                state["should_capture"] = False 

                # 修正时间偏移 (防止看图的时候仿真时间还在后台流逝，导致机械臂突然跳变)
                start_time += (time.time() - step_start)

            if state["ik_enabled"]:

                # 获取物体和末端位置。用鼠标拖动方块，机械臂也能跟着动
                obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
                obj_pos = data.xpos[obj_body_id].copy() # 物体中心
                
                tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
                tcp_pos = data.site(tcp_site_id).xpos
                
                # 默认控制量
                target_pos = tcp_pos.copy() # 默认目标是保持当前位置
                gripper_cmd = GRIPPER_OPEN  # 默认夹爪状态

                # --- 状态机流程 ---
                phase = state["phase"]
                # 阶段 A: 接近 (移动到物体正上方)
                if phase == "APPROACH":
                    target_pos = obj_pos.copy()
                    target_pos[2] += Z_HOVER # 保持高度
                    gripper_cmd = GRIPPER_OPEN
                 

                    # 判断是否到达 (水平误差 < 2cm)
                    if np.linalg.norm(tcp_pos[:2] - target_pos[:2]) < 0.02:
                        state["phase"] = "DESCEND"
                        print(">> 状态切换: 下降")
                # 阶段 B: 下降 (垂直下落到抓取点)
                elif phase == "DESCEND":
                    target_pos = obj_pos.copy()
                    target_pos[2] = obj_pos[2] + Z_GRASP
                    gripper_cmd = GRIPPER_OPEN
                    
                    # 判断是否到达高度
                    if np.linalg.norm(tcp_pos - target_pos) < 0.005:
                        state["phase"] = "GRASP"
                        state["phase_start_time"] = time.time() # 开始计时
                        print(">> 状态切换: 抓取")

                # 阶段 C: 抓取 (闭合夹爪)
                elif phase == "GRASP":
                    target_pos = obj_pos.copy()
                    target_pos[2] = obj_pos[2] + Z_GRASP
                    gripper_cmd = GRIPPER_CLOSE
                    
                    # 等待 0.8 秒让夹爪闭合
                    if time.time() - state["phase_start_time"] > 0.8:
                        state["phase"] = "LIFT"
                        # 记录这一刻的XY坐标，抬起时只动Z轴，防止物体晃动
                        state["lift_target_pos"] = tcp_pos.copy() 
                        state["lift_target_pos"][2] = 0.4 # 设定抬起高度
                        print(">> 状态切换: 抬起")

                # 阶段 D: 抬起
                elif phase == "LIFT":
                    target_pos = state["lift_target_pos"]
                    gripper_cmd = GRIPPER_CLOSE
                    
                    # 如果由于某种原因物体掉落(没抓稳)，这里可以加判断重置回APPROACH
                    
                    if np.linalg.norm(tcp_pos - target_pos) < 0.01:
                        # 任务完成，可以在这里选择悬停或者重新开始
                        pass

                # --- 执行控制 ---
                # 计算 6D IK (位置 + 姿态)
                # 注意：这里需要确保你定义了 solve_6d_ik 函数
                delta_q = solve_6d_ik(model, data, target_pos, target_quat_down)
                
                # 2. 限制速度 (防止瞬移)
                delta_q = np.clip(delta_q, -0.1, 0.1) # 限制每步最大转动 0.1 弧度
                
                # 3. 应用关节角度
                data.ctrl[:6] = data.qpos[:6] + delta_q
                
                # 4. 应用夹爪控制
                data.ctrl[6] = gripper_cmd
                data.ctrl[7] = gripper_cmd

            else:
                # --- 手动模式 ---
                pass 

            # === 物理步进 ===
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # === 时间同步 ===
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)



if __name__ == "__main__":
    model, data = load_ur5()
    
    if model and data:
        init_scene(model, data)
        run_simulation(model, data)