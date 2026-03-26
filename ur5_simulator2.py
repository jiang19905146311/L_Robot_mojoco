'''
ur5_simulation.ur5_simulator1 的 Docstring
单机器人抓取仿真+视觉窗口
'''
import os
# 配置后端
os.environ['MUJOCO_GL'] = 'glfw'

import mujoco
import mujoco.viewer
import time
import numpy as np
import cv2

# === 导入自定义模块 ===
import utils_robot
from utils_vision import VisionModule

# === 全局配置 ===
XML_PATH = "/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/scene_catch.xml"
CAMERA_NAME = "wrist_camera"

# 控制参数
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.04
Z_HOVER = 0.15
Z_GRASP = 0.025
TARGET_QUAT_DOWN = np.array([0, 1, 0, 0]) # 末端垂直向下

def run_simulation():
    # 加载模型
    model, data = utils_robot.load_model(XML_PATH)
    if not model: return

    # 初始化场景
    utils_robot.reset_robot(model, data, GRIPPER_OPEN)

    # 检查相机
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME) == -1:
        print(f"❌ 错误: XML中找不到相机 {CAMERA_NAME}")
        return

    # 状态变量
    state = {
        "ik_enabled": False,        # 是否开启自动抓取
        "cv_enabled": False,         # 是否开启视觉窗口
        "phase": "APPROACH",        # 抓取阶段
        "phase_start_time": 0,      # 阶段计时器
        "lift_target_pos": None     # 抬起目标点缓存
    }

    # 按键回调
    def key_callback(keycode):
        if keycode == 49: # Key '1'
            state["ik_enabled"] = not state["ik_enabled"]
            print(f"🕹️ IK模式: {state['ik_enabled']}")
            if state["ik_enabled"]: state["phase"] = "APPROACH"
            
        elif keycode == 52: # Key '4'
            state["cv_enabled"] = not state["cv_enabled"]
            print(f"👁️ 视觉窗口: {state['cv_enabled']}")
            if not state["cv_enabled"]: cv2.destroyAllWindows()

    # === 启动仿真主循环 ===
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        
        # 延迟初始化渲染器 (防止上下文冲突)
        renderer = mujoco.Renderer(model, height=480, width=640)
        vision_module = VisionModule(width=640, height=480, fovy=60)  # 根据实际相机参数调整
        # 调整上帝视角
        viewer.cam.distance = 2.0
        viewer.cam.lookat = [0.5, 0, 0.5]
        viewer.cam.azimuth = 130
        
        print("✅ 仿真就绪！[1]切换IK [4]开关视觉")

        while viewer.is_running():
            step_start = time.time()
            
            # --- 物理步进 ---
            mujoco.mj_step(model, data)

            # --- 视觉处理模块 ---
            if state["cv_enabled"]:
                try:
                    renderer.update_scene(data, camera=CAMERA_NAME)
                    img_rgb = renderer.render()
                    
                    # 调用 vision_utils
                    res_img, center = vision_module.detect_green_cube(img_rgb)

                    
                    if res_img is not None:
                        cv2.imshow("Wrist Camera", res_img)
                        cv2.waitKey(1)
                except Exception as e:
                    print(f"Vision Error: {e}")
                    state["cv_enabled"] = False

            # --- 机器人控制模块 (IK 状态机) ---
            if state["ik_enabled"]:
                # 获取各个坐标
                obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
                obj_pos = data.xpos[obj_id].copy()
                tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
                tcp_pos = data.site(tcp_id).xpos
                
                target_pos = tcp_pos.copy()
                gripper_cmd = GRIPPER_OPEN

                # 简单状态机
                phase = state["phase"]
                if phase == "APPROACH":
                    target_pos = obj_pos.copy()
                    target_pos[2] += Z_HOVER
                    if np.linalg.norm(tcp_pos[:2] - target_pos[:2]) < 0.02: state["phase"] = "DESCEND"
                
                elif phase == "DESCEND":
                    target_pos = obj_pos.copy()
                    target_pos[2] = obj_pos[2] + Z_GRASP
                    if np.linalg.norm(tcp_pos - target_pos) < 0.005:
                        state["phase"] = "GRASP"
                        state["phase_start_time"] = time.time()
                
                elif phase == "GRASP":
                    target_pos = obj_pos.copy()
                    target_pos[2] = obj_pos[2] + Z_GRASP
                    gripper_cmd = GRIPPER_CLOSE
                    if time.time() - state["phase_start_time"] > 0.8:
                        state["phase"] = "LIFT"
                        state["lift_target_pos"] = tcp_pos.copy()
                        state["lift_target_pos"][2] = 0.4
                
                elif phase == "LIFT":
                    target_pos = state["lift_target_pos"]
                    gripper_cmd = GRIPPER_CLOSE

                # 调用 robot_utils 计算 IK
                delta_q = utils_robot.solve_ik(model, data, target_pos, TARGET_QUAT_DOWN)
                delta_q = np.clip(delta_q, -0.1, 0.1) # 速度限幅
                
                # 应用控制
                data.ctrl[:6] = data.qpos[:6] + delta_q
                data.ctrl[6] = gripper_cmd
                data.ctrl[7] = gripper_cmd

            # --- 同步 ---
            viewer.sync()
            
            # 保持实时性
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_simulation()