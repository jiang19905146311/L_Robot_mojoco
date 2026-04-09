'''
标定实验记录器：
1. 拦截控制台 print 输出，实时写入 TXT 报告
2. 提供统一的接口，将原始矩阵数据打包保存为 NPZ 文件
'''
import sys
import os
import datetime
import numpy as np

class DoubleLogger(object):
    """
    实验记录器：支持 with 语法，自动拦截控制台输出并保存矩阵为 .npy 文件。
    所有内容统一存放在以时间戳命名的子文件夹内。
    """
    def __init__(self, log_dir="logs"):
        # 自动创建 logs 文件夹
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 生成统一的时间戳后缀
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建一个独立子文件夹存放所有的数据文件
        self.data_dir = os.path.join(log_dir, f"calib_data_{timestamp}")

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # 文本报告路径
        self.report_filename = os.path.join(self.data_dir, f"calib_report_{timestamp}.txt")

        # 劫持系统终端输出
        self.terminal = sys.stdout
        self.log_file = open(self.report_filename, "w", encoding='utf-8')
        sys.stdout = self  # 开始拦截

    def __enter__(self):
        """当使用 with DoubleLogger(...) as logger: 时触发"""
        sys.stdout = self  # 正式开始拦截控制台
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """当离开 with 块（无论正常结束还是报错）时触发"""
        self.close()
        # 返回 True，程序报错时不中断
        # 返回 None，报错正常弹出，但日志已经安全保存
        return None

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush() # 实时落盘

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        """恢复系统默认输出，并关闭文件句柄"""
        if sys.stdout == self:
            sys.stdout = self.terminal
        
        if not self.log_file.closed:
            print(f"\n📁 实验记录已完整保存至目录: {self.data_dir}")
            self.log_file.close()


    def save_calibra_raw_data_offline(self, samples, X, Y, Z):
        """
        将采集到的原始数据保存，方便日后脱离仿真器直接调试数学算法
        """
        # 提取数据
        data_dict = {
            "matrix_A": np.array([s['A'] for s in samples]),
            "matrix_B": np.array([s['B'] for s in samples]),
            "matrix_C": np.array([s['C'] for s in samples]),
            "result_X": X,
            "result_Y": Y,
            "result_Z": Z
        }

        # 循环保存为独立的 npy 文件
        for name, data in data_dict.items():
            save_path = os.path.join(self.data_dir, f"{name}.npy")
            np.save(save_path, data)
            
        print(f"\n💾 [数据备份] 已将 {len(data_dict)} 个矩阵成功保存至目录: {self.data_dir}")