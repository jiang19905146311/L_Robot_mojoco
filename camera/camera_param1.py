import os
import sys
import time
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBError, OBFormat, VideoStreamProfile

def main():
    try:
        pipeline = Pipeline()
        config = Config()

        # 1. 尝试获取深度流配置
        try:
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            print(f"🔹 成功匹配深度分辨率: {depth_profile.get_width()}x{depth_profile.get_height()}")
        except OBError as e:
            print(f"⚠️ 无法开启深度流，但我们将继续尝试彩色流: {e}")

        # 2. 尝试获取彩色流配置
        try:
            color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            # 这里我们不用 get_default，而是直接捞出第一个有效的视频流配置，确保它能跟硬件对齐
            color_profile = color_profiles.get_stream_profile_by_index(0).as_video_stream_profile()
            config.enable_stream(color_profile)
            print(f"🔴 成功匹配彩色分辨率: {color_profile.get_width()}x{color_profile.get_height()}")
        except OBError as e:
            print(f"❌ 无法获取彩色流配置: {e}")
            return

        # 3. 开启 Pipeline
        print("⚡ 正在启动相机流并加载硬件标定参数...")
        pipeline.start(config)
        
        # 4. 🔥 关键：强制等待 1~2 秒，等硬件芯片和数据流完全同步完毕
        time.sleep(1.5)
        
        # 5. 连续抓取几帧，确保数据送达 SDK 缓存
        for i in range(10):
            frames = pipeline.wait_for_frames(200)
            if frames is not None:
                break
        
        # 6. 获取参数
        camera_param = pipeline.get_camera_param()

        if camera_param is not None:
            # 兼容性读取
            if hasattr(camera_param, 'rgb_intrinsic'):
                c_intr = camera_param.rgb_intrinsic
                c_dist = camera_param.rgb_distortion
            else:
                c_intr = camera_param.color_intrinsic
                c_dist = camera_param.color_distortion

            print("\n" + "="*40)
            print("🎉 成功读取到真实相机参数！")
            print("="*40)
            print(f"========== 彩色相机内参 ({c_intr.width}x{c_intr.height}) ==========")
            print(f"fx = {c_intr.fx:.4f}")
            print(f"fy = {c_intr.fy:.4f}")
            print(f"cx = {c_intr.cx:.4f}")
            print(f"cy = {c_intr.cy:.4f}")
            print("\n========== 彩色相机畸变参数 ==========")
            print(f"k1 = {c_dist.k1:.6f}")
            print(f"k2 = {c_dist.k2:.6f}")
            print(f"k3 = {c_dist.k3:.6f}")
            print(f"p1 = {c_dist.p1:.6f}")
            print(f"p2 = {c_dist.p2:.6f}")
            
            # 如果全是 0 的备用方案说明
            if c_intr.fx == 0.0:
                print("\n⚠️ 警告：读出来的依然是 0。这说明该设备在出厂时可能未将标定参数写入该特定分辨率下。")
                print("💡 建议：可以尝试更换 `get_stream_profile_by_index(0)` 里的索引（比如改用 1, 2, 3），切换成 1920x1080 试试。")

        pipeline.stop()
        
    except OBError as e:
        print(f"❌ SDK 运行错误: {e}")

if __name__ == "__main__":
    main()