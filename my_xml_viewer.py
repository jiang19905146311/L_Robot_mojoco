import mujoco
import mujoco.viewer
import time

'''
scene_catch.xml
scene_dualcalib.xml

'''

# 加载模型
try:
    model = mujoco.MjModel.from_xml_path("/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/scene_dualcalib.xml")
    data = mujoco.MjData(model)
except ValueError as e:
    print(f"模型加载失败，请检查路径: {e}")
    exit()

print("正在打开 Mujoco 内置可视化窗口...")

# 启动 Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 获取仿真步长，用于控制循环频率
    dt = model.opt.timestep
    
    while viewer.is_running():
        step_start = time.time()

        # 物理步进（如果只想静态看图，可以换回 mj_forward）
        mujoco.mj_step(model, data)

        # 同步 GUI 渲染
        viewer.sync()

        # 简单的频率控制，防止 CPU 占用过高
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


'''python3 -m mujoco.viewer --mjcf=/home/jt/WorkSpace/Mujoco/ur5_simulation/xml_files/scene2.xml'''