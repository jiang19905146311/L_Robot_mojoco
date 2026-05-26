'''
示例代码展示了如何使用 Orbbec Gemini 2 相机进行手眼标定的三维位置解算。
它通过 OpenCV 的 findCirclesGrid 函数检测非对称圆点阵列标定板，并使用 solvePnP 算法计算标定板相对于相机的三维位置 (X, Y, Z)。
代码中还包含了一个简单的滑动平均滤波器来平滑输出的位姿数据，减少视觉抖动。
此外，用户可以按 's' 键保存当前的位姿数据到 JSON 文件中，方便后续分析和使用。
请根据你的实际标定板规格和相机内参进行相应的配置调整。
核心功能包括：
1. 自动扫描可用的摄像头端口，确保能找到 Orbbec Gemini 2 的视频流。
2. 使用自适应二值化来处理图像，增强圆点的对比度，适应不同的光照条件。
3. 通过一系列的过滤条件（面积、长宽比、圆形度、凸性）来筛选出真正的圆点，剔除杂质。
4. 使用 OpenCV 的 findCirclesGrid 函数来识别标定板上的圆点阵列，并在画面上绘制检测结果。
5. 通过 solvePnP 算法求解标定板相对于相机的三维位置 (X, Y, Z)，并在画面上实时显示和绘制三维坐标轴。
6. 提供一个实时的调试窗口，显示过滤后的圆点和二值化结果，帮助你调整参数以获得最佳的检测效果。
7. 支持用户通过按键来退出程序，确保资源能够正确释放。
8. 支持用户按 's' 键保存当前的位姿数据到 JSON 文件中，方便后续分析和使用。
请确保在运行此代码之前已经正确连接了 Orbbec Gemini 2 相机，并安装了 pyorbbecsdk 和 OpenCV 库。

'''
import cv2
import numpy as np
import sys
import signal
import json
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBError

running = True

def handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handler)

def main():
    global running

    # =================================================================
    # 【配置 1】根据你的标定板修改参数
    # =================================================================
    # 标定板阵列规格
    board_size = (4, 5)  # 根据你成功 LOCKED 的尺寸填写，这里以 (4, 5) 为例
    
    # 圆点中心距：两个相邻圆心的真实物理距离（单位：米，手眼标定推荐用米作统一单位）
    # 假设你的标定板圆心距是 15 毫米，则写 0.015
    square_size = 0.040

    # 根据非对称网格定义标定板的「三维世界坐标点」(Z轴默认为0)
    object_points = []
    for i in range(board_size[1]):      
        for j in range(board_size[0]):  
            x = (2 * j + i % 2) * square_size
            y = i * square_size
            object_points.append([x, y, 0])
    object_points = np.array(object_points, dtype=np.float64)

    # =================================================================
    # 【配置 2】相机内参（必须准确，否则输出的三维坐标是错的）
    #  Gemini 2 可以通过 SDK 获取内参，这里先填入一组标准 1080P 虚拟内参供测试
    # =================================================================

    # 初始化 Orbbec Camera SDK 
    pipeline = Pipeline()
    config = Config()

    # 采集数据列表
    captured_data = []
    pose_count = 0

    try:
        # 获取 1920x1080, 30fps, RGB 格式的流配置
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB, 30)
        config.enable_stream(color_profile)
        
        # ⚠️ 必须先 start 开启流，才能获取到绑定了 1080P 分辨率的真实内参
        pipeline.start(config)
        print("✅ Orbbec Gemini 2 启动成功！正在拉取彩色视频流...")
        
        # =================================================================
        # 🌟 自动获取相机的真实物理内参和畸变参数
        # =================================================================
        param = pipeline.get_camera_param()
        
        # 构建 OpenCV 所需的 3x3 内参矩阵
        camera_matrix = np.array([
            [param.rgb_intrinsic.fx, 0.0, param.rgb_intrinsic.cx],
            [0.0, param.rgb_intrinsic.fy, param.rgb_intrinsic.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # 构建 OpenCV 所需的 1x5 畸变系数矩阵 [k1, k2, p1, p2, k3]
        dist_coeffs = np.array([
            param.rgb_distortion.k1, 
            param.rgb_distortion.k2, 
            param.rgb_distortion.p1, 
            param.rgb_distortion.p2, 
            param.rgb_distortion.k3
        ], dtype=np.float64)

        print("\n🎯 成功加载相机真实内参:")
        print(f"fx={param.rgb_intrinsic.fx:.2f}, fy={param.rgb_intrinsic.fy:.2f}")
        print(f"cx={param.rgb_intrinsic.cx:.2f}, cy={param.rgb_intrinsic.cy:.2f}\n")

    except OBError as e:
        print(f"❌ 相机启动失败！请检查连线。{e}")
        sys.exit(1)
    
    # Blob检测器用于提升 findCirclesGrid 内部查找Mask上圆点的性能和准确度
    pure_params = cv2.SimpleBlobDetector_Params()
    pure_detector = cv2.SimpleBlobDetector_create(pure_params)

    cv2.namedWindow("Orbbec Camera Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Orbbec Camera Detection", 900, 800)
    cv2.resizeWindow("debug", 900, 800)


    # 滤波器和状态变量
    smoothed_tvec = None
    smoothed_rvec = None
    # 滑动平均滤波器的平滑因子,取值范围 [0, 1]值越小越平滑但延迟越高
    alpha = 0.5  

    try:
        while running:
            try:
                frames = pipeline.wait_for_frames(100)
            except Exception:
                break

            if frames is None: continue
            color_frame = frames.get_color_frame()
            if color_frame is None: continue

            width = color_frame.get_width()
            height = color_frame.get_height()

            # 将奥比 SDK 返回的数据转换为 OpenCV 格式的 BGR 图像    
            raw_data = color_frame.get_data()
            image_rgb = np.copy(np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3)))
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            found = False

            # --- 步骤 A: 自适应二值化 ---
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 51, 5
            )

            # --- 步骤 B: 严格轮廓筛选 ---
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(gray)
            custom_keypoints = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 800 or area > 35000: continue
                
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.8 or aspect_ratio > 1.2: continue
                    
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity < 0.85: continue
                
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                solidity = float(area) / hull_area
                if solidity < 0.98: continue
                    
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = float(M["m10"] / M["m00"])
                    cY = float(M["m01"] / M["m00"])
                    radius = int(max(w, h) / 2)
                    cv2.circle(clean_mask, (int(cX), int(cY)), radius, 255, -1)
                    kp = cv2.KeyPoint(cX, cY, float(max(w, h)))
                    custom_keypoints.append(kp)

            # 生产白底黑点的「完美标定板」图像，供 findCirclesGrid 查找圆点位置用
            perfect_board = cv2.bitwise_not(clean_mask)

            # --- 步骤 C: 组网与位置解算 ---
            # 只有当我们在图像中找到了足够的圆点（至少15个）时，才尝试进行组网
            if len(custom_keypoints) >= 15:
                ret, centers = cv2.findCirclesGrid(
                    perfect_board, board_size, 
                    flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                    blobDetector=pure_detector
                )
                if ret:
                    
                    cv2.drawChessboardCorners(frame, board_size, centers, ret)
                    found = True
                    
                    # 🚀【核心功能】通过 PnP 算法求解三维位置 (输入内参、标定板物理坐标、当前图像坐标)
                    retval, rvec, tvec = cv2.solvePnP(
                        object_points, centers, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if retval:
                        # EMA滤波器平滑位置和姿态，减少坐标和3D轴的视觉抖动
                        if smoothed_tvec is None:
                            smoothed_tvec = tvec
                            smoothed_rvec = rvec
                        else:
                            smoothed_tvec = alpha * tvec + (1 - alpha) * smoothed_tvec
                            smoothed_rvec = alpha * rvec + (1 - alpha) * smoothed_rvec

                        # tvec 是平移向量，代表标定板中心点在相机坐标系下的 X, Y, Z (单位：米)
                        X, Y, Z = smoothed_tvec[0][0], smoothed_tvec[1][0], smoothed_tvec[2][0]

                        # 终端打印输出位置
                        # 大量刷屏输出
                        #print(f"📍 标定板三维位置(米) -> X: {X:+.4f}m, Y: {Y:+.4f}m, Z: {Z:+.4f}m")
                        # 使用 \r 回车符实现同一行更新输出，避免大量日志刷屏
                        print(f"\r📍 标定板三维位置(米) -> X: {X:+.4f}m, Y: {Y:+.4f}m, Z: {Z:+.4f}m",end="")
                        
                        # 在画面上实时绘制三维坐标轴 (红轴=X, 绿轴=Y, 蓝轴=Z)，方便肉眼验证
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, smoothed_rvec, smoothed_tvec, 0.05) # 轴长 5 厘米
                        
                        # UI 界面文字显示位置
                        cv2.putText(frame, f"X: {X:.3f} Y: {Y:.3f} Z: {Z:.3f} m", 
                                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # UI 绘制与显示
            if found:
                cv2.putText(frame, "LOCKED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                # 丢失目标时，清空历史滤波数据
                smoothed_tvec = None
                smoothed_rvec = None
                cv2.putText(frame, "Searching...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

            # Debug 图像拼装显示
            frame_debug = cv2.drawKeypoints(frame.copy(), custom_keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            h_small, w_small = int(height/4), int(width/4)
            thresh_small = cv2.resize(thresh, (w_small, h_small))
            # 阈值图像放在右下角显示
            frame_debug[height-h_small:height, width-w_small:width] = cv2.cvtColor(thresh_small, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Orbbec Camera Detection", frame)
            cv2.imshow("debug", frame_debug)

            key = cv2.waitKey(1) & 0xFF

            # 按下 ESC 键退出
            if not running or key == 27:
                break
            # 按下 's' 键保存当前的位姿数据到列表中（前提是当前已经成功 LOCKED）
            elif key == ord('s') and found:
                pose_count += 1

                R, _ = cv2.Rodrigues(smoothed_rvec)
                T_matrix = np.eye(4)
                T_matrix[:3, :3] = R
                T_matrix[:3, 3] = smoothed_tvec.reshape(3)

                # 记录数据
                data_point = {
                    "pose_id": pose_count,
                    "tvec": smoothed_tvec.flatten().tolist(),
                    "rvec": smoothed_rvec.flatten().tolist(),
                    "T_cam_to_board": T_matrix.tolist()
                }
                captured_data.append(data_point)
                print(f"\n💾 已保存第 {pose_count} 组位姿数据点")

                with open("calibration_data.json", "w") as f:
                    json.dump(captured_data, f, indent=4)


    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()