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

        # 仿真相机无“鱼眼”或“桶形”畸变，全设置为0
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
    
    def init_kalman(self):
        '''
        初始化卡尔曼滤波器，用于平滑位姿估计结果，减少视觉检测的噪声影响。
        状态量包括位置和速度，共12维；测量量只有位置和角度，共6维。
        '''
        # 状态量有6个：x, y, z, roll, pitch, yaw
        # 加上它们的速度，共12个变量
        self.kf = cv2.KalmanFilter(12, 6) 
        
        # 状态转移矩阵 A (假设匀速运动)
        self.kf.transitionMatrix = np.eye(12, dtype=np.float32)
        dt = 1/30.0  # 假设你的相机是 30 FPS
        for i in range(6):
            self.kf.transitionMatrix[i, i+6] = dt
            
        # 测量矩阵 H (我们只能直接观测到位置和角度，观测不到速度)
        self.kf.measurementMatrix = np.zeros((6, 12), np.float32)
        for i in range(6):
            self.kf.measurementMatrix[i, i] = 1
            
        # 过程噪声 Q (相信预测多一点，还是相信模型多一点)
        self.kf.processNoiseCov = np.eye(12, dtype=np.float32) * 1e-4
        
        # 测量噪声 R (视觉检测的误差权重，越大越平滑，但延迟越高)
        self.kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * 1e-2

    def detect_and_pose(self, img_rgb):
        """
        位姿估计
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

        # 初始化返回值：即使没找到也返回统一格式，防止外部解包报错
        success = False
        rvec, tvec, R_mat = None, None, None
        img_points = None  # 存储 2D 点
        obj_points = None  # 存储 3D 点

        # 在图中寻找棋盘格角点，使用增强型标志位
        # CALIB_CB_ADAPTIVE_THRESH: 自适应阈值，自适应二值化，应对光照不均
        # CALIB_CB_FAST_CHECK: 快速检查，如果没有棋盘格则迅速跳过
        # CALIB_CB_NORMALIZE_IMAGE: 图像归一化
        # CALIB_CB_FILTER_QUADS: 过滤四边形，提高准确率

        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH 
         + cv2.CALIB_CB_FAST_CHECK 
         + cv2.CALIB_CB_NORMALIZE_IMAGE 
         + cv2.CALIB_CB_FILTER_QUADS)

        # found 是布尔值；corners 是检测到的 2D 像素坐标
        found, corners = cv2.findChessboardCorners(img_gray, self.pattern_size, flags)
        
        if found:
            # 亚像素优化
            # 初始检测出的像素点可能不够准，用算法在 11x11 的窗口内寻找更精确的亚像素位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
            corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

            # 保存 2D 和 3D 对应点
            # corners2 的形状通常是 (N, 1, 2)，为了方便后续计算，我们 reshape 成 (N, 2)
            img_points = corners2.reshape(-1, 2)
            obj_points = self.objp.reshape(-1, 3)

            # PnP (Perspective-n-Point) 解算
            # 根据已知的 3D 点 (objp) 和对应的 2D 像素点 (corners2)
            # 结合相机内参 (K)，算出标定板相对于相机的 3D 变换
            # ret: 求解是否成功；rvec: 旋转向量；tvec: 平移向量（即 xyz 坐标）
            ret, rvec, tvec = cv2.solvePnP(self.objp, corners2, self.K, self.dist_coeffs)
            # [修改] 加入 flags=cv2.SOLVEPNP_IPPE，使用专为平面棋盘格设计的算法，大幅减少180度翻转报错
            # ret, rvec, tvec = cv2.solvePnP(self.objp, corners2, self.K, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
            # ret, rvec, tvec, inliers= cv2.solvePnPRansac(self.objp, corners2, self.K, self.dist_coeffs,reprojectionError=2.0, iterationsCount=100)

            if ret and rvec is not None and rvec.shape[0] > 0:

                success = True
                
                # 转换旋转向量为旋转矩阵 (3x3)
                # 在控制机械臂时，旋转矩阵或四元数比旋转向量更好用
                R_mat, _ = cv2.Rodrigues(rvec)

                # 在图上把找出来的角点连成线画出来
                cv2.drawChessboardCorners(img_bgr, self.pattern_size, corners2, found)
                # 绘制 3D 坐标轴并添加文字标注位置信息（长度 0.1m）。红色是 X，绿色是 Y，蓝色是 Z，线条粗细3
                cv2.drawFrameAxes(img_bgr, self.K, self.dist_coeffs, rvec, tvec, 0.1, 3)
                
                # 在图上实时打印距离信息 (单位: m)
                dist_text = f"Pos: x={tvec[0][0]:.2f} y={tvec[1][0]:.2f} z={tvec[2][0]:.2f}"
                cv2.putText(img_bgr, dist_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 将所有科研、标定、分析需要的数据全部打包
        # 返回：是否成功、旋转向量、平移向量、画好图的图像

        return {
            "success": success,
            "rvec": rvec,
            "tvec": tvec,
            "R_mat": R_mat,
            "img_points": img_points,  # 用于算重投影误差
            "obj_points": obj_points,  # 用于算重投影误差
            "img_bgr": img_bgr         # 用于显示或保存
        }


    def detect_and_pose_with_kalman(self, img_rgb):
        # --- 预测 (Prediction) ---
        # 即使没看到目标，卡尔曼也会给出一个预测值
        prediction = self.kf.predict()
        
        # 执行你之前的检测逻辑...
        success, rvec, tvec, R_mat, img_pts, obj_pts, img_draw = self.detect_and_pose_v2(img_rgb)
        
        if success:
            # --- 更新 (Update) ---
            # 将旋转向量 rvec 转为欧拉角，方便滤波（也可直接滤 rvec）
            # 这里为了简单，直接构造测量向量 [x, y, z, r, p, y]
            measurement = np.array([
                tvec[0], tvec[1], tvec[2], 
                rvec[0], rvec[1], rvec[2]
            ], dtype=np.float32).reshape(6, 1)
            
            # 修正滤波器
            estimated = self.kf.correct(measurement)
        else:
            # 如果这帧丢了，直接用预测值维持状态
            estimated = prediction

        # --- 提取平滑后的位姿 ---
        smooth_tvec = estimated[0:3].reshape(3, 1)
        smooth_rvec = estimated[3:6].reshape(3, 1)
        
        # 用平滑后的位姿重新生成旋转矩阵或画图
        smooth_R, _ = cv2.Rodrigues(smooth_rvec)
        
        return success, smooth_rvec, smooth_tvec, smooth_R, img_draw


