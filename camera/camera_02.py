'''
Orbbec Gemini 2 标定板检测示例
这个脚本使用 Orbbec SDK 和 OpenCV 来检测标定板上的圆点，并且在画面上显示检测结果。
你可以通过调整自适应二值化的参数，或者增加更多的过滤条件，来适应不同的环境光照和标定板设计。
记得在运行之前安装 pyorbbecsdk 和 opencv-python 包，并且确保 Orbbec Gemini 2 已经正确连接到电脑。
这个脚本的核心功能包括：
1. 自动扫描可用的摄像头端口，确保能找到 Orbbec Gemini 2 的视频流。
2. 使用自适应二值化来处理图像，增强圆点的对比度，适应不同的光照条件。
3. 通过一系列的过滤条件（面积、长宽比、圆形度、凸性）来筛选出真正的圆点，剔除杂质。
4. 使用 OpenCV 的 findCirclesGrid 函数来识别标定板上的圆点阵列，并在画面上绘制检测结果。
5. 提供一个实时的调试窗口，显示过滤后的圆点和二值化结果，帮助你调整参数以获得最佳的检测效果。
6. 支持用户通过按键来退出程序，确保资源能够正确释放。 
            
'''

import cv2
import numpy as np
import sys
import signal
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBError

# 全局变量，用于控制程序的运行状态
running = True

def handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handler)

def main():

    global running

    # 初始化 Orbbec Pipeline 和 Config
    pipeline = Pipeline()
    config = Config()

    try:
        # 配置彩色数据流：获取 XxY 的 RGB 图像，帧率 30
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB, 30)
        config.enable_stream(color_profile)
        
        # 启动相机流
        pipeline.start(config)
        print("✅ Orbbec Gemini 2 启动成功！正在拉取彩色视频流...")
        
    except OBError as e:
        print(f"❌ 相机启动失败！请检查连线。报错信息: {e}")
        sys.exit(1)


    # 创建一个极简的、不做任何限制的斑点检测器，专门用来读取我们过滤后的完美画面
    pure_params = cv2.SimpleBlobDetector_Params()

    # 面积
    pure_params.filterByArea = False
    # 圆度过滤
    pure_params.filterByCircularity = False
    # 凸性过滤（圆点通常是凸包）
    pure_params.filterByConvexity = False
    # 惯性比过滤（防止太扁）
    pure_params.filterByInertia = False

    pure_detector = cv2.SimpleBlobDetector_create(pure_params)

    # 可能的阵列规格
    possible_sizes = [(3, 7), (7, 3),  (4, 4),  (4, 5), (5, 4)]

    print("\n==================================")
    print("🎥 画面已就绪！将标定板对准镜头。")
    print("👉 按 'Esc 键退出程序")
    print("==================================")

    # 创建一个名字叫 "Orbbec Camera Detection" 的特殊窗口
    # cv2.WINDOW_NORMAL 允许我们用代码改变窗口大小，或者用鼠标手动拖拽改变大小
    cv2.namedWindow("Orbbec Camera Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    
    # 强行把这个窗口的物理显示大小限制在 （或者 640x480）
    # 这样它在屏幕上就很小巧，但相机的核心数据依然是 1080P 的高画质！
    cv2.resizeWindow("Orbbec Camera Detection", 900, 800)
    cv2.resizeWindow("debug", 900, 800)


    try:
        while running:
            # 等待获取一帧数据 (超时时间 100ms)
            try:
                frames = pipeline.wait_for_frames(100)
            except Exception:
                break

            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            # 将 Orbbec 数据转换为 OpenCV 的 numpy 数组
            width = color_frame.get_width()
            height = color_frame.get_height()

            # 使用 frombuffer + reshape 替代 resize，确保像素完美对齐
            raw_data = color_frame.get_data()
            image_rgb = np.copy(np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3)))
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 标定板检测逻辑
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = False
            detected_size = None

            # --- 自适应二值化：完美对抗环境光照不均和反光 ---
            # 如果你的标定板是 白底黑点：用 cv2.THRESH_BINARY_INV (黑点变白，方便找轮廓)
            # 如果你的标定板是 黑底白点：用 cv2.THRESH_BINARY
 
            thresh = cv2.adaptiveThreshold(
                    gray,        # 输入：灰度图（必须是黑白灰图，不能是彩色）
                    255,         # 最大值：超过阈值的像素变成 255（白色）
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 方法：高斯加权（更精准、抗噪）
                    cv2.THRESH_BINARY,               # 模式：亮变白，暗变黑
                    55,          # 区块大小：55×55 的小窗口（决定局部计算范围）
                    5            # 常数C：微调亮度（值越大，黑点越多）
            )

            # --- 寻找所有轮廓 ---
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建一张纯黑画布，用来绘制过滤后的干净圆点
            clean_board = np.zeros_like(gray)
            custom_keypoints = []

            for cnt in contours:
                # 过滤条件 1：面积不能太小也不能太大 (1080P画面下，圆点通常在 100~20000 像素之间)
                area = cv2.contourArea(cnt)
                if area < 200 or area > 15000:
                    continue
                
                # 过滤条件 2：通过外接矩形的长宽比，过滤掉杂质（圆的外接矩形长宽比接近 1:1）
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.75 or aspect_ratio > 1.25: # 允许大角度倾斜导致的椭圆
                    continue

                # 过滤条件3. 增加真·圆形度过滤
                # 圆形度公式： 4 * pi * 面积 / (周长^2)，完美的圆结果为 1.0
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity < 0.6: # 只要不是圆或者椭圆，直接扔掉
                    continue

                # 过滤条件4. 增加凸性过滤
                # 计算凸包面积与轮廓面积的比值，越完美越圆，比值越接近 1.0
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                solidity = float(area) / hull_area # 这个比值叫“实心度”，对非圆剔除有奇效
                if solidity < 0.98: # 只要稍微不规则，直接扔
                    continue
                    
                # 计算这个轮廓的质心 (Center of Mass)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = float(M["m10"] / M["m00"])
                    cY = float(M["m01"] / M["m00"])

                    # 在黑画布上画一个实心的白圆，吐给 OpenCV 识别
                    radius = int(max(w, h) / 2)
                    cv2.circle(clean_board, (int(cX), int(cY)), radius, 255, -1)
                    
                    kp = cv2.KeyPoint(cX, cY, size=float(max(w, h)))
                    custom_keypoints.append(kp)

            # 因为 OpenCV 检测黑点，我们将干净的画布反色成白底黑点
            clean_board_inv = cv2.bitwise_not(clean_board)

            # --- 绘制 DEBUG 画面 ---
            # 此时 frame_debug 显示的是我们通过自定义规则筛选出来的所有点
            frame_debug = cv2.drawKeypoints(frame.copy(), custom_keypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # 把二值化后的黑白画面也贴到 debug 窗口的右上角，方便看有没有“破洞”
            h_small, w_small = int(height/4), int(width/4)
            thresh_small = cv2.resize(thresh, (w_small, h_small))
            thresh_color = cv2.cvtColor(thresh_small, cv2.COLOR_GRAY2BGR)
            frame_debug[0:h_small, width-w_small:width] = thresh_color

            cv2.putText(frame_debug, f"Custom Detected Blobs: {len(custom_keypoints)}", 
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- 组网识别 ---
            # 通过判断点的数量或直接尝试匹配
            if len(custom_keypoints) >= 15: # 只有当画面中的点足够多时才尝试组网，提升效率
                for size in possible_sizes:
                    # 注意：不传 blobDetector，OpenCV 会在内部直接对传入的某种特征组网
                    # 或者我们可以直接使用更高级的查找机制
                    ret, centers = cv2.findCirclesGrid(
                        clean_board_inv, size,
                        flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                        blobDetector=pure_detector 
                    )
                    if ret:
                        cv2.drawChessboardCorners(frame, size, centers, ret)
                        found = True
                        detected_size = size
                        break

            # UI 绘制
            if found:
                cv2.putText(frame, f"LOCKED! Size: {detected_size}", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Searching for board...", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

            # 显示画面
            cv2.imshow("Orbbec Camera Detection", frame)
            cv2.imshow("debug", frame_debug)


            # 检查退出条件，如果用户按了 'q' 键或者外部信号触发了退出，跳出循环
            if not running:
                break
            # 按 27 是 ESC 键
            if cv2.waitKey(1) & 0xFF in [27]:
                running = False
                break
                
    except KeyboardInterrupt:
        pass

    # 捕获其他可能的异常，确保程序不会崩溃，并且能正常关闭相机流和窗口
    finally:
        try:
            pipeline.stop()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # 确保窗口完全关闭

        except Exception as e:
            print("pipeline stop failed:", e)



if __name__ == "__main__":
    main()