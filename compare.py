import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载你的双臂模型
XML_PATH = "/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/scene_dualcalib.xml"  # 确保文件名正确

def save_point_cloud_to_pcd(points, filename, label=None):
    """
    将NumPy点云保存为PCD文件（ASCII格式）
    points: 点云数组，形状为 (N, 3)，每行是[x,y,z]
    filename: 保存的文件名（如 "robot1_workspace.pcd"）
    label: 可选，点的标签/颜色值（用于区分不同机械臂）
    """
    # 获取点云数量
    n_points = points.shape[0]
    
    # 构建PCD文件头（ASCII格式标准头）
    pcd_header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        f"FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {n_points}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n_points}",
        "DATA ascii"
    ]
    
    # 如果需要添加标签/颜色字段（可选）
    if label is not None:
        pcd_header[2] = "FIELDS x y z label"
        pcd_header[3] = "SIZE 4 4 4 4"
        pcd_header[4] = "TYPE F F F I"
        pcd_header[5] = "COUNT 1 1 1 1"
        # 拼接点坐标和标签
        points_with_label = np.hstack([points, label.reshape(-1, 1)])
        points_to_save = points_with_label
    else:
        points_to_save = points
    
    # 写入文件
    with open(filename, 'w') as f:
        # 写入头信息
        for line in pcd_header:
            f.write(line + '\n')
        # 写入点云数据（保留6位小数）
        np.savetxt(f, points_to_save, fmt='%.6f')
    
    print(f"点云已保存为: {filename}")

def get_workspace_cloud(model, data, robot_prefix, n_samples=100000):
    """
    生成指定机器人的工作空间点云
    robot_prefix: 关节名称后缀，例如 "" (Robot1) 或 "_2" (Robot2)
    """
    points = []
    
    # 获取关节的 ID 和 范围
    joint_names = [
        f"shoulder_pan_joint{robot_prefix}",
        f"shoulder_lift_joint{robot_prefix}",
        f"elbow_joint{robot_prefix}",
        f"wrist_1_joint{robot_prefix}",
        f"wrist_2_joint{robot_prefix}",
        f"wrist_3_joint{robot_prefix}"
    ]
    
    # 获取 TCP 的 ID
    tcp_name = "tcp" if robot_prefix == "" else "tcp_2"
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_name)
    
    # 获取关节索引和限位
    q_indices = []
    q_limits = []
    for j_name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        q_addr = model.jnt_qposadr[j_id]
        q_indices.append(q_addr)
        q_limits.append(model.jnt_range[j_id])
    
    q_limits = np.array(q_limits)
    
    print(f"正在计算 {tcp_name} 的采样点...")
    
    # 随机采样
    for _ in range(n_samples):
        # 在限位内随机生成角度
        rand_q = np.random.uniform(q_limits[:, 0], q_limits[:, 1])
        
        # 设置状态
        for i, idx in enumerate(q_indices):
            data.qpos[idx] = rand_q[i]
            
        # 正运动学计算
        mujoco.mj_kinematics(model, data)
        
        # 记录 TCP 世界坐标
        points.append(data.site(tcp_id).xpos.copy())
        
    return np.array(points)

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    # 生成 Robot 1 的点云 (蓝色)
    cloud1 = get_workspace_cloud(model, data, "", n_samples=10000)
    
    # 生成 Robot 2 的点云 (红色)
    cloud2 = get_workspace_cloud(model, data, "_2", n_samples=10000)
    
    # === 保存点云为PCD文件 ===
    # 保存Robot1点云（标签1）
    save_point_cloud_to_pcd(cloud1, "robot1_workspace.pcd", label=np.ones(len(cloud1)))
    # 保存Robot2点云（标签2）
    save_point_cloud_to_pcd(cloud2, "robot2_workspace.pcd", label=np.ones(len(cloud2))*2)
    # 保存合并后的总点云（标签0表示混合）
    all_cloud = np.vstack([cloud1, cloud2])
    all_labels = np.hstack([np.ones(len(cloud1)), np.ones(len(cloud2))*2])
    save_point_cloud_to_pcd(all_cloud, "dual_arm_workspace.pcd", label=all_labels)
    
    # === 修复坐标轴比例，避免椭圆变形 ===
    # 计算统一的坐标轴范围
    x_min, y_min, z_min = np.min(all_cloud, axis=0)
    x_max, y_max, z_max = np.max(all_cloud, axis=0)
    max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) / 2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    
    # === 绘图 ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制采样点
    ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], c='b', s=1, alpha=0.1, label='Robot 1 (Camera)')
    ax.scatter(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2], c='r', s=1, alpha=0.1, label='Robot 2 (Board)')
    
    # 绘制基座位置作为参考
    ax.scatter(0, -0.5, 0, c='k', marker='^', s=100, label='Base 1')
    ax.scatter(0, 0.5, 0, c='k', marker='^', s=100, label='Base 2')
    
    # 设置坐标轴比例一致 + 统一显示范围（解决椭圆问题）
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Dual Arm Workspace Intersection')
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    main()



