import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

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

def get_workspace_cloud(model, data, robot_prefix, n_samples=1000000):
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
    
    # 容错：检查TCP是否存在
    if tcp_id == -1:
        raise ValueError(f"未找到Site: {tcp_name}，请检查XML文件！")
    
    # 获取关节索引和限位
    q_indices = []
    q_limits = []
    for j_name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id == -1:
            raise ValueError(f"未找到关节: {j_name}，请检查XML文件！")
        q_addr = model.jnt_qposadr[j_id]
        q_indices.append(q_addr)
        q_limits.append(model.jnt_range[j_id])
    
    q_limits = np.array(q_limits)
    
    print(f"正在计算 {tcp_name} 的采样点...")
    
    # 批量生成随机角度（优化效率）
    rand_qs = np.random.uniform(q_limits[:, 0], q_limits[:, 1], size=(n_samples, 6))
    
    # 随机采样
    for rand_q in rand_qs:
        # 设置状态
        for i, idx in enumerate(q_indices):
            data.qpos[idx] = rand_q[i]
            
        # 正运动学计算
        mujoco.mj_kinematics(model, data)
        
        # 记录 TCP 世界坐标
        points.append(data.site(tcp_id).xpos.copy())
        
    return np.array(points)


def get_bounding_box(points):
    """计算点云的包围盒（最小/最大坐标）"""
    if len(points) == 0:
        return None, None
    return np.min(points, axis=0), np.max(points, axis=0)

def calculate_intersection(cloud1, cloud2, threshold=0.01):
    """
    计算两个点云的交集（距离阈值法）
    threshold: 两点距离小于该值则认为是交集点（单位：米）
    """
    # 先过滤出cloud1中在cloud2包围盒内的点（缩小计算范围）
    bb2_min, bb2_max = get_bounding_box(cloud2)
    mask = np.all((cloud1 >= bb2_min - threshold) & (cloud1 <= bb2_max + threshold), axis=1)
    cloud1_in_bb2 = cloud1[mask]

    # KDTree加速最近邻搜索
    kdtree = KDTree(cloud2)
    distances, _ = kdtree.query(cloud1_in_bb2)
    return cloud1_in_bb2[distances < threshold]

def draw_compact_bbox(ax, points, color='g', linewidth=2, alpha=0.8):
    """绘制点云的紧凑包围盒（只框住实际点的范围）"""
    bb_min, bb_max = get_bounding_box(points)
    if bb_min is None or bb_max is None:
        print("无交集点，无法绘制紧凑包围盒！")
        return
    
    # 定义立方体8个顶点
    x = [
        bb_min[0], bb_max[0], bb_max[0], bb_min[0],
        bb_min[0], bb_max[0], bb_max[0], bb_min[0]
    ]
    y = [
        bb_min[1], bb_min[1], bb_max[1], bb_max[1],
        bb_min[1], bb_min[1], bb_max[1], bb_max[1]
    ]
    z = [
        bb_min[2], bb_min[2], bb_min[2], bb_min[2],
        bb_max[2], bb_max[2], bb_max[2], bb_max[2]
    ]
    
    # 定义立方体12条棱边
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)   # 竖边
    ]
    
    # 绘制所有棱边
    for edge in edges:
        ax.plot3D(
            [x[edge[0]], x[edge[1]]],
            [y[edge[0]], y[edge[1]]],
            [z[edge[0]], z[edge[1]]],
            color=color, linewidth=linewidth, alpha=alpha,
            label='Compact Intersection BBox' if edge == (0,1) else ""
        )

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    # 生成 Robot 1 的点云 (蓝色)
    cloud1 = get_workspace_cloud(model, data, "", n_samples=100000)
    
    # 生成 Robot 2 的点云 (红色)
    cloud2 = get_workspace_cloud(model, data, "_2", n_samples=100000)
    
    # === 计算精准的交集点云 ===
    intersection_cloud = calculate_intersection(cloud1, cloud2, threshold=0.01)

    # === 保存点云为PCD文件 ===
    # 保存Robot1点云（标签1）
    save_point_cloud_to_pcd(cloud1, "/home/jt/WorkSpace/Mujoco/ur5_simulation/pcd_files/robot1_workspace.pcd", label=np.ones(len(cloud1)))
    # 保存Robot2点云（标签2）
    save_point_cloud_to_pcd(cloud2, "/home/jt/WorkSpace/Mujoco/ur5_simulation/pcd_files/robot2_workspace.pcd", label=np.ones(len(cloud2))*2)
    # 保存合并后的总点云（标签0表示混合）
    all_cloud = np.vstack([cloud1, cloud2])
    all_labels = np.hstack([np.ones(len(cloud1)), np.ones(len(cloud2))*2])
    save_point_cloud_to_pcd(all_cloud, "/home/jt/WorkSpace/Mujoco/ur5_simulation/pcd_files/dual_arm_workspace.pcd", label=all_labels)


    # -----统一显示范围 -----
    # 合并所有点云，计算整体包围盒
    all_cloud = np.vstack([cloud1, cloud2])
    x_min, y_min, z_min = np.min(all_cloud, axis=0)
    x_max, y_max, z_max = np.max(all_cloud, axis=0)
    
    # 计算三个轴的最大跨度，统一显示范围（保证1:1:1视觉比例）
    max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) / 2
    # 计算每个轴的中心
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    # ----------------------

    # === 计算交集和包围盒 ===
    # 计算各自包围盒
    bb1_min, bb1_max = get_bounding_box(cloud1)
    bb2_min, bb2_max = get_bounding_box(cloud2)
  
    # 计算整体交集包围盒
    intersect_min = np.max([bb1_min, bb2_min], axis=0)
    intersect_max = np.min([bb1_max, bb2_max], axis=0)

    # 计算交集点云的包围盒（更紧凑）
    bb3_min, bb3_max = get_bounding_box(intersection_cloud)
    
    # 计算交集点云（量化共同可达区域）
    intersection_cloud2 = calculate_intersection(cloud1, cloud2, threshold=0.02)  # 2cm阈值，适当调整以平衡精度和计算量
    
    # 打印量化结果
    print("\n==== 工作空间分析结果 ====")
    print(f"Robot1 工作空间范围：")
    print(f"  X: [{bb1_min[0]:.3f}, {bb1_max[0]:.3f}] m")
    print(f"  Y: [{bb1_min[1]:.3f}, {bb1_max[1]:.3f}] m")
    print(f"  Z: [{bb1_min[2]:.3f}, {bb1_max[2]:.3f}] m")
    print(f"\nRobot2 工作空间范围：")
    print(f"  X: [{bb2_min[0]:.3f}, {bb2_max[0]:.3f}] m")
    print(f"  Y: [{bb2_min[1]:.3f}, {bb2_max[1]:.3f}] m")
    print(f"  Z: [{bb2_min[2]:.3f}, {bb2_max[2]:.3f}] m")
    
    if np.all(intersect_min < intersect_max):
        print(f"\n工作空间交集范围：")
        print(f"  X: [{intersect_min[0]:.3f}, {intersect_max[0]:.3f}] m")
        print(f"  Y: [{intersect_min[1]:.3f}, {intersect_max[1]:.3f}] m")
        print(f"  Z: [{intersect_min[2]:.3f}, {intersect_max[2]:.3f}] m")
        print(f"交集点数量：{len(intersection_cloud2)}")
    else:
        print("\n⚠️  两个机械臂工作空间无交集！")

    if np.all(bb3_min < bb3_max):
        print(f"\n交集点云的紧凑包围盒范围：")
        print(f"  X: [{bb3_min[0]:.3f}, {bb3_max[0]:.3f}] m")
        print(f"  Y: [{bb3_min[1]:.3f}, {bb3_max[1]:.3f}] m")
        print(f"  Z: [{bb3_min[2]:.3f}, {bb3_max[2]:.3f}] m")
        print(f"交集点数量：{len(intersection_cloud)}")
    else:
        print("\n⚠️ 交集点云无效，无法计算紧凑包围盒！")

    # === 绘图优化 ===
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原始点云（降低透明度，避免遮挡）
    ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], c='b', s=1, alpha=0.1, label='Robot 1 (Camera)')
    ax.scatter(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2], c='r', s=1, alpha=0.1, label='Robot 2 (Board)')
    
    # 绘制交集点云（绿色，突出显示）
    if len(intersection_cloud) > 0:
        ax.scatter(intersection_cloud[:, 0], intersection_cloud[:, 1], intersection_cloud[:, 2], 
                   c='g', s=5, alpha=0.5, label='Intersection (Common Area)')
    
    # 绘制基座位置（增大尺寸，更醒目）
    ax.scatter(0, -0.5, 0, c='k', marker='^', s=100, label='Base 1')
    ax.scatter(0, 0.5, 0, c='k', marker='^', s=100, label='Base 2')
    
    # 绘制交集点云的紧凑包围盒（只框住有实际点的区域）
    if len(intersection_cloud) > 0:
        draw_compact_bbox(ax, intersection_cloud)

    if np.all(intersect_min < intersect_max):
        # 定义立方体8个顶点的坐标（按固定顺序：底面4个点 → 顶面4个点）
        # 顶点顺序：[左下后, 右下后, 右下前, 左下前, 左下后顶, 右下后顶, 右下前顶, 左下前顶]
        x = [
            intersect_min[0], intersect_max[0], intersect_max[0], intersect_min[0],
            intersect_min[0], intersect_max[0], intersect_max[0], intersect_min[0]
        ]
        y = [
            intersect_min[1], intersect_min[1], intersect_max[1], intersect_max[1],
            intersect_min[1], intersect_min[1], intersect_max[1], intersect_max[1]
        ]
        z = [
            intersect_min[2], intersect_min[2], intersect_min[2], intersect_min[2],
            intersect_max[2], intersect_max[2], intersect_max[2], intersect_max[2]
        ]
        
        # 定义立方体12条棱的顶点索引（每两个索引对应一条棱）
        edges = [
            # 底面4条棱（连接底面4个点）
            (0, 1), (1, 2), (2, 3), (3, 0),
            # 顶面4条棱（连接顶面4个点）
            (4, 5), (5, 6), (6, 7), (7, 4),
            # 4条竖边（连接底面和顶面对应点）
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # 遍历所有棱，绘制线条
        for edge in edges:
            ax.plot3D(
                [x[edge[0]], x[edge[1]]],
                [y[edge[0]], y[edge[1]]],
                [z[edge[0]], z[edge[1]]],
                'g-', linewidth=2, alpha=0.8, label='Intersection BBox' if edge == (0,1) else ""
            )



    # === 统一坐标轴显示范围 ===
    ax.set_box_aspect([1,1,1])  # 保证轴的物理长度比例
    # 强制设置每个轴的显示范围为「中心±最大跨度」，确保1m视觉长度一致
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    # 优化坐标轴（增大字体、调整刻度）
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)  # 增大刻度字体
    ax.set_title('Dual Arm Workspace & Intersection Analysis', fontsize=14, pad=20)
    
    # 优化图例（增大字体、调整位置）
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
