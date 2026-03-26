'''
相机-标定板标定
'''
import mujoco
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import os

# -------------------------- 从 XML 文件加载模型 --------------------------
print("正在加载 main.xml (包含 checkerboard.xml)...")
model = mujoco.MjModel.from_xml_path("/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/scene_checkerboard.xml")
data = mujoco.MjData(model)

# 提高渲染分辨率，避免锯齿导致角点模糊
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)

# -------------------------- 标定参数配置 --------------------------
# 必须与 checkerboard.xml 中的 texrepeat="7 6" 对应
# 内部角点数 = (列数-1, 行数-1)
BOARD_COLS = 6  # 7 - 1
BOARD_ROWS = 5  # 6 - 1
SQUARE_SIZE = 0.05 

# 准备标定数据容器
obj_points = [] # 世界坐标系中的点
img_points = [] # 图像坐标系中的点

# 生成标定板的局部 3D 坐标
objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE

# 获取对象 ID
try:
    board_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "checkerboard")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_cam")
except Exception as e:
    print(f"ID获取失败: {e}")
    print("请检查 XML 中 body name 是否为 'checkerboard', camera name 是否为 'fixed_cam'")
    exit()

# -------------------------- 数据采集循环 --------------------------
print("开始采集图像... (按 'q' 键可提前退出)")
valid_frames = 0
target_frames = 15
rng = np.random.default_rng(42)

while valid_frames < target_frames:
    # 随机生成位置 (根据你的相机位置微调了范围)
    # 左右，前后，上下 XYZ（m）
    rand_pos = np.array([
        rng.uniform(-0.1, 0.1), # x: 左右移动
        rng.uniform(0, 0.05),  # y: 深度
        rng.uniform(0.15, 0.45)     # z: 上下高度
    ])
    
    # 随机生成旋转 (四元数)
    # 俯仰，偏航，滚动 RPY（度）
    rx = rng.uniform(-15, 15)
    ry = rng.uniform(-15, 15)
    rz = rng.uniform(-10, 10)
    quat = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_quat() 
    # Scipy (x,y,z,w) -> Mujoco (w,x,y,z)
    mujoco_quat = np.array([quat[3], quat[0], quat[1], quat[2]])

    # 应用位置和旋转
    model.body_pos[board_id] = rand_pos
    model.body_quat[board_id] = mujoco_quat
    
    # 更新物理状态
    mujoco.mj_forward(model, data)

    # --- 渲染图像 ---
    renderer.update_scene(data, camera="fixed_cam")
    img_rgb = renderer.render()
    
    # 转换颜色空间
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # 用于显示
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)   # 用于检测

    # 增加 flag: ADAPTIVE_THRESH (自适应阈值) 和 NORMALIZE_IMAGE (归一化)
    # 这对光照不均匀或对比度低的情况非常有效
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    # --- 检测棋盘格 ---
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), flags)

    debug_img = img_bgr.copy() # 复制一份用于画图，保持原图干净（可选）
    
    if ret:
        # 亚像素精确化
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # 存入列表
        obj_points.append(objp)
        img_points.append(corners2)
        valid_frames += 1
        
        # 绘制角点
        cv2.drawChessboardCorners(debug_img, (BOARD_COLS, BOARD_ROWS), corners2, ret)
        status_text = f"Success: {valid_frames}/{target_frames}"
        color = (0, 255, 0)
    else:
        status_text = "Searching..."
        color = (0, 0, 255)

    # 在图像上打印状态
    cv2.putText(debug_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 缩小显示以便查看 (如果屏幕不够大)
    display_img = cv2.resize(debug_img, (640, 360))

    # --- 显式显示窗口 ---
    cv2.imshow('Calibration View', debug_img)
    
    # 等待时间，按 'q' 退出
    key = cv2.waitKey(300)

    if key & 0xFF == ord('q'):
        print("用户强制停止采集。")
        break

cv2.destroyAllWindows()

# -------------------------- 执行标定计算 --------------------------
if valid_frames > 0:
    print("\n采集结束，正在计算相机参数...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    print("\n========== 标定结果 ==========")
    print(f"重投影误差 (RMS): {ret:.4f}")
    print("\n[内参矩阵 K]:")
    print(mtx)
    print("\n[畸变系数][k1, k2, p1, p2, k3]:")
    print(dist.ravel())
else:
    print("未采集到有效图像，无法标定。")


