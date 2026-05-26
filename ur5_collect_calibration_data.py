'''
这个脚本的目的是为 UR5 机器人采集一组安全的 TCP 位姿和对应的关节角数据，以用于后续的机器人建模和校准。
它分为两个主要步骤：1. 预筛选安全位姿2. 执行安全点位采集
在预筛选阶段，我们随机生成大量的关节角组合，并使用机器人的正向运动学（FK）功能计算对应的 TCP 位姿。
然后，我们通过一系列简单的规则（如高度限制、水平范围限制和工作半径限制）来过滤掉那些可能导致机器人碰撞或超出工作范围的位姿。
在执行采集阶段，我们连接到机器人并依次移动到预筛选出的安全位姿，记录下实际的关节角和 TCP 位姿数据。
最后，我们将这些数据保存到本地文件中，以供后续分析和建模使用。

'''

import time
import numpy as np
from datetime import datetime
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface

# ==========================================
# 配置
# ==========================================
ROBOT_IP = "192.168.134.100"

TOTAL_CANDIDATES = 500  # 预生成数量
REQUIRED_SAMPLES = 50   # 最终需要的安全组数

# TCP 安全立方体限制 (单位: mm)
Z_MIN, Z_MAX = 10, 1100    # 防止撞桌面 (Z=0通常是底座平面)
X_MIN, X_MAX = -800, 800
Y_MIN, Y_MAX = -800, 800
MAX_RADIUS = 1200           # 离底座中心的最大距离

# 设置安全的关节活动范围 (根据你的实际工作台环境调整！防止撞机！)
# 顺序: [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
Q_MIN = np.array([-np.pi/2, -np.pi*3/4, -np.pi/2, -np.pi*3/4, -np.pi, -np.pi])
Q_MAX = np.array([ np.pi/2,        0.0,  np.pi/2,  0.0,   np.pi,  np.pi])

# ==========================================
# 辅助函数：简单位姿过滤
# ==========================================
def is_pose_safe(tcp_pose):
    """
    检查 TCP 位姿是否在安全空间内
    tcp_pose 格式: [x, y, z, rx, ry, rz]
    """
    x, y, z = tcp_pose[:3]
    x_mm, y_mm, z_mm = x * 1000, y * 1000, z * 1000  # 转换为毫米单位
    radius = np.sqrt(x_mm**2 + y_mm**2)
    
    # 规则 1: 高度限制
    if not (Z_MIN <= z_mm <= Z_MAX):
        return False
    # 规则 2: 水平范围限制
    if not (X_MIN <= x_mm <= X_MAX and Y_MIN <= y_mm <= Y_MAX):
        return False
    # 规则 3: 工作半径限制 (防止伸太直)
    if radius > MAX_RADIUS:
        return False
    
    return True

# ==========================================
# 第一步：预筛选安全位姿
# ==========================================
print("Filtering safe poses...")
# 临时连接机器人以利用其内部的正向运动学(FK)计算功能
# 如果不想连机器人，也可以用第三方库如 'forward_kinematics'
rtde_c = RTDEControlInterface(ROBOT_IP)

safe_q_list = []
attempts = 0

while len(safe_q_list) < REQUIRED_SAMPLES and attempts < TOTAL_CANDIDATES:
    attempts += 1
    # 随机生成一组关节角
    test_q = np.random.uniform(Q_MIN, Q_MAX).tolist()
    
    # 使用机器人的 FK 接口计算该关节角下的 TCP 位置
    # 注意：getForwardKinematics 不会移动机器人
    test_tcp = rtde_c.getForwardKinematics(test_q)
    
    if is_pose_safe(test_tcp):
        safe_q_list.append(test_q)

print(f"Filter done. Found {len(safe_q_list)} safe poses from {attempts} candidates.")

# ==========================================
# 第二步：执行安全点位采集
# ==========================================
rtde_r = RTDEReceiveInterface(ROBOT_IP)
collected_q = []
collected_tcp = []

try:
    for i, q in enumerate(safe_q_list):
        print(f"Moving to sample {i+1}/{len(safe_q_list)}...")
        
        # 执行移动（角度，速度，加速度）
        rtde_c.moveJ(q, 0.4, 0.4)
        time.sleep(0.5)
        
        # 记录实际值
        collected_q.append(rtde_r.getActualQ())
        collected_tcp.append(rtde_r.getActualTCPPose())

finally:
    today_date = datetime.now().strftime("%Y.%m.%d")
    np.save(f"{today_date}calibration_q.npy", np.array(collected_q))
    np.save(f"{today_date}calibration_tcp.npy", np.array(collected_tcp))
    rtde_c.stopScript()
    print("💾 Data saved. Connection closed.")


