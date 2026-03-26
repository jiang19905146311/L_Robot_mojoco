'''
utils_vision.VisionModule 的 Docstring
视觉模块，包含相机内参配置、图像处理算法（如颜色检测、棋盘格检测）以及位姿估计功能。
三个坐标系：标定板坐标系（Z轴垂直向上），相机坐标系（Z轴沿光轴指向前方），世界坐标系（仿真中的全局坐标系）。
'''
import cv2
import numpy as np

class VisionModule:
    def __init__(self, width, height, fovy):
        # 保存图像的宽和高（像素单位）
        self.width = width
        self.height = height
        
        # --- 计算焦距 f ---
        # fovy 是垂直视场角（弧度制）。
        # 公式：f = (图像高度的一半) / tan(视角的一半)，f = (h / 2) / tan(fovy / 2)
        # 这步是将角度转化为像素焦距，决定了物体的成像缩放比例。
        f = (height / 2) / np.tan(np.deg2rad(fovy) / 2)

        # 计算相机内参矩阵 K
        # [fx, 0,  cx]  fx, fy 是焦距
        # [0,  fy, cy]  cx, cy 是图像中心点（主点）
        # [0,  0,  1 ]
        self.K = np.array([
            [f, 0, width / 2],
            [0, f, height / 2],
            [0, 0, 1]
        ])
        # 仿真相机无畸变，全设置为0
        self.dist_coeffs = np.zeros(5) 

        # --- 棋盘格配置 ---
        # 这里的 (7, 5) 指的是棋盘格内部“角点”的数量（行 7 个，列 5 个）需要根据棋盘格调整！
        self.pattern_size = (7, 5)
        # 现实中每个方格的物理尺寸（单位：米），这里是 5cm(XML总宽0.35m / 7)
        self.square_size = 0.05  

        # --- 构造 3D 标定板坐标系 ---
        # 3D 世界坐标系下的角点 (Z=0)，标定板平面的 Z 轴全为 0
        # 生成一个矩阵，用来存储角点在标定板自己坐标系下的位置
        self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        # 用 mgrid 生成网格坐标 (0,0), (1,0)... (6,4)，然后转置铺平
        self.objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        # 乘以实际格长，把索引坐标变成物理长度坐标（米）
        self.objp *= self.square_size

    def detect_green_cube(self, img_rgb):
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

    def calculate_error(self, current_center, image_center=(320, 240)):
        # 计算当前物体中心和画面中心的偏差
        ex = current_center[0] - image_center[0]
        ey = current_center[1] - image_center[1]
        return ex, ey

    def detect_and_pose(self, img_rgb):
        """
        返回:
            found: 是否找到
            rvec, tvec: 标定板相对于相机的位姿
            img_draw: 画了轴的图
        """
        # 拷贝图像防止内存报错
        # 避免 OpenCV 直接修改 MuJoCo 传来的原始只读内存导致报错
        # cvtColor 将 RGB 转为 BGR，OpenCV 的函数默认处理 BGR
        img_bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
        # 转成单通道灰度图寻找角点
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 在图中寻找棋盘格角点
        # found 是布尔值（找到了吗？）；corners 是检测到的 2D 像素坐标
        found, corners = cv2.findChessboardCorners(img_gray, self.pattern_size, None)

        # 先初始化结果，万一没找到就返回空
        rvec, tvec = None, None
        
        if found:
            # 亚像素优化
            # 初始检测出的像素点可能不够准，用算法在 11x11 的窗口内寻找更精确的亚像素位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

            # PnP (Perspective-n-Point) 解算
            # 根据已知的 3D 点 (objp) 和对应的 2D 像素点 (corners2)
            # 结合相机内参 (K)，算出标定板相对于相机的 3D 变换
            # ret: 求解是否成功；rvec: 旋转向量；tvec: 平移向量（即 xyz 坐标）
            ret, rvec, tvec = cv2.solvePnP(self.objp, corners2, self.K, self.dist_coeffs)

            # 在图上把找出来的角点连成线画出来
            cv2.drawChessboardCorners(img_bgr, self.pattern_size, corners2, found)

            if ret:
                # 绘制 3D 坐标轴（长度 0.1m）。红色是 X，绿色是 Y，蓝色是 Z
                img_bgr = cv2.drawFrameAxes(img_bgr, self.K, self.dist_coeffs, rvec, tvec, 0.1)

        # 返回：是否成功、旋转向量、平移向量、画好图的图像
        return found, rvec, tvec, img_bgr