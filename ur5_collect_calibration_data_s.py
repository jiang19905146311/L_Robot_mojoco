'''
# 这个脚本的目的是在 MuJoCo 中预览随机生成的关节位姿对应的末端位姿，以便你在正式采集数据前确认模型和坐标系设置正确。
# 请确保你的 XML 模型文件中末端执行器（TCP）的位置和朝向设置正确，与你的实际采集脚本一致。
# 运行这个脚本后，你应该能在 MuJoCo 的可视化窗口中看到 UR 机器人在不同的随机位姿下的末端位置，确认它们是否在合理范围内。
# 如果你看到末端位置在预期范围内，并且没有明显的错误，那么你就可以放心地运行正式的采集脚本了。  
'''

import mujoco
import mujoco.viewer
import numpy as np
import time

# ==========================================
# 1. 这里的参数必须与你采集脚本中的一致SS
# ==========================================
Q_MIN = np.array([-np.pi/2, -np.pi*3/4, -np.pi/4, -np.pi*3/4, -np.pi, -np.pi])
Q_MAX = np.array([ np.pi/2,        0,  np.pi/4,  0.0,   np.pi,  np.pi])
XML_PATH = "/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/ur5.xml"
NUM_SAMPLES = 20  # 预览 20 组即可

# 加载 UR 机器人模型 (请确保你有模型文件，或者使用内置示例)
model = mujoco.MjModel.from_xml_path(XML_PATH) 
data = mujoco.MjData(model)

print("Starting MuJoCo Visualization...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(NUM_SAMPLES):
        # 模拟生成随机位姿
        target_q = np.random.uniform(Q_MIN, Q_MAX)
        
        # 将关节角写入仿真器 (前6个通常是关节，取决于模型定义)
        data.qpos[:6] = target_q
        
        # 计算正向运动学
        mujoco.mj_forward(model, data)
        
        # 刷新视图
        viewer.sync()
        
        print(f"Checking Sample {i+1}/{NUM_SAMPLES}: {np.round(target_q, 2)}")
        
        # 停留一秒让你观察
        time.sleep(1.0)
        
        if not viewer.is_running():
            break

print("Preview finished.")