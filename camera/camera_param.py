import os
import sys
from datetime import datetime
from pyorbbecsdk import Pipeline, OBSensorType, OBError, Config, OBFormat, VideoStreamProfile

def main():

    # ==========================================
    # 配置：在这里指定你想保存的目标文件夹路径
    # ==========================================
    # 方案 A (相对路径): 放在当前目录下的 camera_info 文件夹中
    TARGET_DIR = "./camera_info" 
    
    # 方案 B (绝对路径): 如果你想放桌面，可以改成下面这样 (取消注释即可)
    # TARGET_DIR = "/home/jt/Desktop/camera_data"

    # 🛠️ 获取当前时间，格式化为：20260518_195030 (年月日_时分秒)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 🛠️ 让文件名动态带上时间戳
    filename = f"camera_profiles_{timestamp}.txt"

    # 自动检测：如果指定的文件夹不存在，代码会自动创建它，防止报错
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📁 检测到目标文件夹不存在，已自动创建: {TARGET_DIR}")

    # 定义输出的文件名
    # 使用 os.path.join 把文件夹路径和文件名安全地拼接在一起
    full_path = os.path.join(TARGET_DIR, filename)

    try:
        pipeline = Pipeline()
        config = Config()

        # ==========================================
        # ⚠️ 必须先配置并开启数据流
        # ==========================================
        # 获取彩色和深度的配置列表
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        # 获取默认配置并启用 (你也可以根据需要指定具体的分辨率)
        color_profile = color_profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # 启动 pipeline，此时相机会根据分辨率计算正确的内参
        pipeline.start(config)
        print("✅ 成功启动相机数据流，正在读取参数...\n")


        # 获取内参和畸变了
        camera_param = pipeline.get_camera_param()

        # 使用 utf-8 编码打开文件准备写入
        # "w"（Write，覆盖写）改成 "a"（Append，追加写）
        with open(full_path, "w", encoding="utf-8") as f:
            
            f.write("==================================================\n")
            f.write("📸 Orbbec Gemini 2 相机信息\n")
            f.write(f"当前分辨率设定 - 彩色: {color_profile.get_width()}x{color_profile.get_height()} | 深度: {depth_profile.get_width()}x{depth_profile.get_height()}\n")
            f.write("==================================================\n\n")

            # ==================================================
            # 彩色相机内参
            # ==================================================
            rgb_intrinsic = camera_param.rgb_intrinsic

            f.write("========== 彩色相机内参 ==========\n")
            f.write(f"fx = {rgb_intrinsic.fx}\n")
            f.write(f"fy = {rgb_intrinsic.fy}\n")
            f.write(f"cx = {rgb_intrinsic.cx}\n")
            f.write(f"cy = {rgb_intrinsic.cy}\n")
            f.write(f"width  = {rgb_intrinsic.width}\n")
            f.write(f"height = {rgb_intrinsic.height}\n\n")

            print("\n========== 彩色相机内参 ==========")
            print(f"fx = {rgb_intrinsic.fx}")
            print(f"fy = {rgb_intrinsic.fy}")
            print(f"cx = {rgb_intrinsic.cx}")
            print(f"cy = {rgb_intrinsic.cy}")

            # ==================================================
            # 彩色相机畸变参数
            # ==================================================
            rgb_distortion = camera_param.rgb_distortion

            f.write("========== 彩色相机畸变参数 ==========\n")
            f.write(f"k1 = {rgb_distortion.k1}\n")
            f.write(f"k2 = {rgb_distortion.k2}\n")
            f.write(f"p1 = {rgb_distortion.p1}\n")
            f.write(f"p2 = {rgb_distortion.p2}\n")
            f.write(f"k3 = {rgb_distortion.k3}\n")
            f.write(f"k4 = {rgb_distortion.k4}\n")
            f.write(f"k5 = {rgb_distortion.k5}\n")
            f.write(f"k6 = {rgb_distortion.k6}\n\n")

            print("\n========== 彩色相机畸变参数 ==========")
            print(f"k1 = {rgb_distortion.k1}")
            print(f"k2 = {rgb_distortion.k2}")
            print(f"p1 = {rgb_distortion.p1}")
            print(f"p2 = {rgb_distortion.p2}")
            print(f"k3 = {rgb_distortion.k3}")

            # ==================================================
            # 深度相机内参
            # ==================================================
            depth_intrinsic = camera_param.depth_intrinsic

            f.write("========== 深度相机内参 ==========\n")
            f.write(f"fx = {depth_intrinsic.fx}\n")
            f.write(f"fy = {depth_intrinsic.fy}\n")
            f.write(f"cx = {depth_intrinsic.cx}\n")
            f.write(f"cy = {depth_intrinsic.cy}\n")
            f.write(f"width  = {depth_intrinsic.width}\n")
            f.write(f"height = {depth_intrinsic.height}\n\n")

            # ==================================================
            # 深度相机畸变
            # ==================================================
            depth_distortion = camera_param.depth_distortion

            f.write("========== 深度相机畸变参数 ==========\n")
            f.write(f"k1 = {depth_distortion.k1}\n")
            f.write(f"k2 = {depth_distortion.k2}\n")
            f.write(f"p1 = {depth_distortion.p1}\n")
            f.write(f"p2 = {depth_distortion.p2}\n")
            f.write(f"k3 = {depth_distortion.k3}\n")
            f.write(f"k4 = {depth_distortion.k4}\n")
            f.write(f"k5 = {depth_distortion.k5}\n")
            f.write(f"k6 = {depth_distortion.k6}\n\n")

            # ==================================================
            # 支持的配置列表
            # ==================================================
            f.write("==================================================\n")
            f.write("📸 Orbbec Gemini 2 彩色相机支持的所有配置列表：\n")
            f.write("==================================================\n")
            f.write(f"总计找到 {color_profile_list.get_count()} 组有效配置：\n\n")

        
            # 遍历并打印每一组配置
            for i in range(color_profile_list.get_count()):
                # 转换为视频流配置对象
                profile = color_profile_list.get_stream_profile_by_index(i)
                
                width = profile.get_width()
                height = profile.get_height()
                fps = profile.get_fps()
                format_enum = profile.get_format() 

                # 安全转换：提取枚举的名称字符串（如 'RGB' 或 'MJPEG'）
                try:
                    format_str = format_enum.name
                except AttributeError:
                    format_str = str(format_enum)

                # 拼接单行信息
                info_line = f"配置 [{i:02d}]: 分辨率={width}x{height} | 帧率={fps}fps | 格式={format_str}\n"
                
                # 同时写入文件并打印到终端
                f.write(info_line)
                print(info_line.strip())
                
        print(f"\n✅ 所有配置已成功保存到本地文本: {full_path}")

        # 读取完成后关闭 pipeline
        pipeline.stop()

    except OBError as e:
        print(f"❌ 读取相机配置失败！请确保相机已正确连接 USB 3.0。错误信息: {e}")

if __name__ == "__main__":
    main()