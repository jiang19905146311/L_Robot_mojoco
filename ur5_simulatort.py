import mujoco
import mujoco.viewer
import glfw
import numpy as np

# -------------------------- 1. 加载Mujoco模型 --------------------------
model_xml = """
<mujoco model="pendulum">
  <option timestep="0.005"/>
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1"/>
    <body pos="0 0 0">
      <joint name="pendulum_hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.5" rgba="0.8 0.2 0.2 1"/>
      <geom type="sphere" size="0.06" pos="0 0 0.5" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="pendulum_hinge" ctrlrange="-1 1" gear="100"/>
  </actuator>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

# -------------------------- 2. 配置自定义相机的参数 --------------------------
# 用于自定义窗口的相机
camera = mujoco.MjvCamera()
camera.azimuth = 90  # 侧视角
camera.elevation = -10
camera.distance = 1.2
camera.lookat = np.array([0.0, 0.0, 0.3])

# 渲染选项
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True # 示例：开启接触点显示

# 扰动对象
perturb = mujoco.MjvPerturb()

# 渲染掩码
catmask = int(mujoco.mjtCatBit.mjCAT_ALL)

# -------------------------- 3. 创建自定义 GLFW 窗口 --------------------------
def create_visual_window(width=640, height=480, title="Custom Camera View"):
    if not glfw.init():
        raise RuntimeError("无法初始化GLFW")
    
    # 请求特定配置以避免兼容性问题
    glfw.window_hint(glfw.SAMPLES, 4) # 4x 抗锯齿
    
    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("无法创建窗口")
    
    # 设为当前上下文
    glfw.make_context_current(window)
    glfw.swap_interval(1) # 开启垂直同步
    
    # 【修复】MjrContext 通常只需要 model 和 fontscale
    # 在这个 Context 初始化时，必须保证 glfw.make_context_current 已经指向了该窗口
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    return window, ctx

visual_window, render_context = create_visual_window(640, 480)
viewport = mujoco.MjrRect(0, 0, 640, 480)

# Scene 对象必须在循环外初始化，避免内存泄漏
# maxgeom 是场景中能容纳的最大几何体数量
scene = mujoco.MjvScene(model, maxgeom=1000)

# -------------------------- 4. 启动主查看器 + 循环 --------------------------
# 启动官方被动查看器（这是一个独立的窗口）
viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.azimuth = 0
viewer.cam.distance = 2.0

try:
    while True:
        # 1. 检查退出条件
        if (glfw.window_should_close(visual_window) or 
            not viewer.is_running() or 
            glfw.get_key(visual_window, glfw.KEY_ESCAPE) == glfw.PRESS):
            break
        
        # 2. 物理仿真步进
        data.ctrl[0] = 0.5 * np.sin(data.time) # 简单的正弦控制
        mujoco.mj_step(model, data)
        
        # 3. 同步官方 Viewer（主窗口）
        viewer.sync()

        # 4. 渲染自定义 GLFW 窗口
        # 必须显式切换上下文，因为 viewer.sync() 可能切换到了内部的上下文
        glfw.make_context_current(visual_window)
        
        # 更新场景数据
        # 注意：这里使用位置参数调用，比 kwargs 更安全兼容
        mujoco.mjv_updateScene(
            model, 
            data, 
            opt, 
            perturb, 
            camera, 
            catmask, 
            scene
        )

        # 执行渲染
        mujoco.mjr_render(viewport, scene, render_context)
        
        # 交换缓冲区并处理事件
        glfw.swap_buffers(visual_window)
        glfw.poll_events()

finally:
    viewer.close()
    glfw.destroy_window(visual_window)
    glfw.terminate()
    print("程序已退出")