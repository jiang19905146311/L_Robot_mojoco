import os
import glob
import mujoco

def get_latest_xml(folder="xml_files"):
    # 获取目录下所有 xml 文件
    files = glob.glob(os.path.join(folder, "*.xml"))
    if not files:
        return None
    # 按修改时间排序，取最新的一个
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

# 自动获取路径
XML_PATH = get_latest_xml()

if XML_PATH:
    print(f"🚀 正在加载最新的模型文件: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
else:
    print("❌ 文件夹里没有找到 XML 文件！")