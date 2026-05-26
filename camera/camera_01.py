'''
初次尝试的摄像头检测脚本，功能如下：
1. 启动时自动扫描 0-9 端口，找出能打开并返回画面的摄像头。
2. 显示第一个可用摄像头的实时画面，并在画面上显示当前端口号。
3. 用户可以按 'n' 键切换到下一个可用摄像头端口，画面会更新显示新的摄像头画面和端口号。
4. 用户可以按 'q' 键退出程序。
5. 在画面上实时检测棋盘格（使用 OpenCV 的 findCirclesGrid 函数），如果检测到棋盘格，会在画面上绘制出检测到的圆点，并显示 "LOCKED!" 提示；如果没有检测到，则显示 "Searching for board..." 提示。
6. 代码中包含了对 OpenCV 版本兼容性的修复，确保在不同环境下都能正常运行。
'''

import cv2
import numpy as np

def scan_available_ports(max_ports=10):
    """扫描所有能打开并返回画面的摄像头端口"""
    print(f"正在扫描 0 到 {max_ports-1} 的视频端口，请稍候...")
    working_ports = []
    for index in range(max_ports):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                working_ports.append(index)
        cap.release()
    return working_ports

def live_detect_interactive():
    # 1. 扫描可用端口
    ports = scan_available_ports()
    if not ports:
        print("❌ 找不到任何有效的视频流！请检查数据线，或关闭其他占用相机的软件。")
        return
    
    print(f"✅ 找到可用端口列表: {ports}")
    
    # ================= 程序刚启动时设置 =================
    current_port_idx = 0
    cap = cv2.VideoCapture(ports[current_port_idx])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # =============================================================

    # 2. 斑点检测器配置 (提高圆点识别率)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 10000
    params.filterByCircularity = True
    params.minCircularity = 0.8
    blobDetector = cv2.SimpleBlobDetector_create(params)

    # 3. 可能的阵列规格 (请注意你的板子很可能是 3x7 或 4x11)
    possible_sizes = [(3, 7), (7, 3), (4, 11), (11, 4), (5, 5)]

    print("\n==================================")
    print("🎥 画面已启动！")
    print("👉 请在视频窗口按 'n' 键切换下一个摄像头端口")
    print("👉 请在视频窗口按 'q' 键退出程序")
    print("==================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败，尝试下一个端口...")
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = False
        detected_size = None
        
        # 尝试检测标定板
        for size in possible_sizes:
            ret_grid, centers = cv2.findCirclesGrid(
                gray, size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=blobDetector
            )
            if ret_grid:
                cv2.drawChessboardCorners(frame, size, centers, ret_grid)
                found = True
                detected_size = size
                break 

        # 在画面上绘制 UI 提示信息
        h, w = frame.shape[:2]
        # 背景半透明黑条，方便看字
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1) 
        
        current_port_real = ports[current_port_idx]
        cv2.putText(frame, f"Port: {current_port_real} | Press 'n' to Switch | 'q' to Quit", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if found:
            cv2.putText(frame, f"LOCKED! Size: {detected_size}", 
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Searching for board...", 
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        # 显示画面
        cv2.imshow('Interactive Camera Detection', frame)

        # 处理键盘按键
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            # 切换到下一个端口
            # ================= 切换相机时设置 =================
            cap.release()
            current_port_idx = (current_port_idx + 1) % len(ports)
            next_port = ports[current_port_idx]
            print(f"🔄 正在切换到端口 {next_port} ...")

            cap = cv2.VideoCapture(next_port)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_detect_interactive()