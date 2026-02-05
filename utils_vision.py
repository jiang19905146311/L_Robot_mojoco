import cv2
import numpy as np

def detect_green_cube(img_rgb):
    """
    输入: MuJoCo 的 RGB 图像
    输出: 标注后的 BGR 图像, 物体中心 (cx, cy)
    """
    # 1. 基础判空
    if img_rgb is None: return None, None
    
    try:
        # ==========================================
        # 核心修复：重建纯净内存，解决 OpenCV 兼容性问题
        # ==========================================
        h, w, c = img_rgb.shape
        img_clean = np.zeros((h, w, c), dtype=np.uint8)
        img_clean[:] = img_rgb[:]
        # ==========================================

        # 转为 BGR (OpenCV 标准)
        img_bgr = cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 绿色阈值 (根据实际光照可能需要微调)
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(img_hsv, lower_green, upper_green)
        
        # 形态学去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            # 过滤太小的噪点
            if cv2.contourArea(c) > 100:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cx = x + w // 2
                cy = y + h // 2
                center = (cx, cy)
                
                cv2.circle(img_bgr, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(img_bgr, f"Pos: {cx},{cy}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
        return img_bgr, center

    except Exception as e:
        print(f"❌ 视觉模块报错: {e}")
        return None, None
    


def calculate_error(current_center, image_center=(320, 240)):
    # 计算当前物体中心和画面中心的偏差
    ex = current_center[0] - image_center[0]
    ey = current_center[1] - image_center[1]
    return ex, ey



def detect_checkerboard(img_rgb, cam_matrix, dist_coeffs, pattern_size=(6, 5), square_size=0.025):
    """
    输入:
        img_rgb: MuJoCo 渲染出的图像
        cam_matrix: 相机内参矩阵 (3x3)
        dist_coeffs: 畸变系数 (MuJoCo 理想相机通常为 0)
        pattern_size: 棋盘格内部角点行列数 (w, h) -> checkerboard.xml 是 7x6 格子，所以角点是 6x5
        square_size: 单个格子的边长 (米) -> checkerboard.xml 中 0.175/7 = 0.025m
    输出:
        img_draw: 绘制了坐标轴的图像
        tvec: 标定板中心相对于相机的平移向量
        rvec: 标定板相对于相机的旋转向量
    """
    if img_rgb is None: return None, None, None

    # 1. 图像预处理
    h, w, c = img_rgb.shape
    img_clean = np.zeros((h, w, c), dtype=np.uint8)
    img_clean[:] = img_rgb[:] # 内存深拷贝，防止 OpenCV 报错
    img_bgr = cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. 寻找棋盘格角点
    found, corners = cv2.findChessboardCorners(img_gray, pattern_size, None)

    if found:
        # 亚像素优化角点坐标
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

        # 3. 定义世界坐标系下的角点位置 (假设标定板平面 z=0)
        # 坐标原点设在棋盘格第一个角点
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp = objp * square_size

        # 4. PnP 解算 (2D -> 3D)
        ret, rvec, tvec = cv2.solvePnP(objp, corners2, cam_matrix, dist_coeffs)

        if ret:
            # 绘制角点
            cv2.drawChessboardCorners(img_bgr, pattern_size, corners2, found)
            # 绘制坐标轴 (长度 0.1m)
            img_bgr = cv2.drawFrameAxes(img_bgr, cam_matrix, dist_coeffs, rvec, tvec, 0.1)
            
            # 显示距离信息
            dist = np.linalg.norm(tvec)
            cv2.putText(img_bgr, f"Dist: {dist:.3f}m", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            return img_bgr, tvec, rvec

    return img_bgr, None, None