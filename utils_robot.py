'''
utils_robot.py 的 Docstring
机器人相关的工具函数，包括几何模型建立、状态重置、逆运动学求解等功能。
'''
import math
import numpy as np
from scipy.optimize import minimize


class AngleUtils:
    #=====================================================
    # 角度转换工具方法
    #=====================================================
    @staticmethod
    def axis_angle_to_quaternion(axis, angle_rad):
        """
        将轴角转换为四元数
        
        :param axis: 旋转轴，形如 [x, y, z] 的列表或元组
        :param angle_rad: 旋转角度（弧度）
        :return: 四元数 [w, x, y, z]
        """
        x, y, z = axis
        # 确保轴向量是单位向量（归一化）
        length = math.sqrt(x**2 + y**2 + z**2)
        if length < 1e-6:
            return [1.0, 0.0, 0.0, 0.0]  # 零向量则返回无旋转四元数
        
        x /= length
        y /= length
        z /= length
        
        half_angle = angle_rad / 2.0
        sin_half = math.sin(half_angle)
        
        w = math.cos(half_angle)
        xq = x * sin_half
        yq = y * sin_half
        zq = z * sin_half
        
        return [w, xq, yq, zq]

    @staticmethod
    def euler_to_quaternion(x, y, z):
        """
        将欧拉角（ZYX 顺规：Roll-Pitch-Yaw）转换为四元数
        
        :param x: 绕 X 轴的旋转弧度 (Pitch 俯仰角)
        :param y: 绕 Y 轴的旋转弧度 (Yaw 偏航角)
        :param z: 绕 Z 轴的旋转弧度 (Roll 翻滚角)
        :return: 四元数 [w, x, y, z]
        """
        # 计算各轴半角的正弦和余弦值
        cx = math.cos(x * 0.5)
        sx = math.sin(x * 0.5)
        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cz = math.cos(z * 0.5)
        sz = math.sin(z * 0.5)

        # ZYX 顺规进行四元数乘法展开后的最终公式
        w = cx * cy * cz + sx * sy * sz
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * cy * sz

        return [w, qx, qy, qz]

class ModelingUtils:
    
    #=====================================================
    # 机械臂几何建模方法
    #=====================================================
    # 使用 @staticmethod，这样调用时不需要实例化类，且不会误传 self 参数
    @staticmethod
    def DH_transform(a, alpha, d, theta):
        """标准 DH 变换矩阵 (Standard DH)"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,      sa,     ca,    d],
            [0,       0,      0,    1]
        ])  

    @staticmethod
    def MDH_transform(a, alpha, d, theta):
        """"改进 DH 变换矩阵 (Modified DH - Craig's convention)"""
        # 注意：MDH 的经典定义中，变换顺序通常是 Rx(alpha_{i-1}) -> Dx(a_{i-1}) -> Rz(theta_i) -> Dz(d_i)
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*sa, ca, d*ca],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def POE_transform(w, v, theta): 
        """计算 POE 指数映射变换矩阵 (Exponential Mapping)"""
        w = np.array(w)
        v = np.array(v)
        
        # 判读是否为纯平移（移动关节），防止 w_hat 为零矩阵导致计算逻辑失效
        if np.linalg.norm(w) < 1e-10:
            T = np.eye(4)
            T[:3, 3] = v * theta
            return T

        w_hat = np.array([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])
        
        w_hat_2 = w_hat @ w_hat
        R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat_2
        # 平移部分 p 的闭式解
        p = (np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat_2) @ v
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

class KinematicsUtils:
    #=====================================================
    # 机械臂运动学方法
    #=====================================================
    @staticmethod
    def forward_kinematics_dh(joint_angles, dh_params, mode='DH'):
        """
        计算基于 DH 参数的正运动学
        mode: 'DH' 或 'MDH' 
        """
        T = np.eye(4)
        for i in range(len(joint_angles)):
            a, alpha, d, theta_offset = dh_params[i]
            if mode == 'DH':
                Ti = ModelingUtils.DH_transform(a, alpha, d, theta_offset + joint_angles[i])
            else:
                Ti = ModelingUtils.MDH_transform(a, alpha, d, theta_offset + joint_angles[i])
            T = T @ Ti
        return T

    @staticmethod
    def forward_kinematics_poe(joint_angles, screw_axes, M):
        """
        基于 POE 的正运动学 (Space Frame)
        T = e^[S1]q1 * e^[S2]q2 * ... * e^[Sn]qn * M
        """
        T = np.eye(4)
        for i in range(len(joint_angles)):
            w, v = screw_axes[i]
            Ti = ModelingUtils.POE_transform(w, v, joint_angles[i])
            T = T @ Ti
        return T @ M



    @staticmethod
    def inverse_kinematics_numerical(target_pose, initial_guess, screw_axes, M, joint_limits=None):
        """
        基于数值优化的逆运动学求解器
        :param target_pose: 4x4 目标齐次变换矩阵
        :param initial_guess: 初始关节角 (List 或 Array)
        :param screw_axes: POE 螺旋轴列表
        :param M: 零位位姿矩阵
        :param joint_limits: 关节限位 [(min, max), ...] 默认为 None
        :return: 优化后的关节角
        """
        
        # 定义目标函数：位姿误差的二范数
        def objective(thetas):
            T_curr = KinematicsUtils.forward_kinematics_poe(thetas, screw_axes, M)
            error_vec = ErrorModelingUtils.matrix_error(T_curr, target_pose)
            # 加权：旋转误差通常比平移误差数值大，这里可以根据需要调整权重
            return np.linalg.norm(error_vec)

        # 执行优化
        res = minimize(
            objective, 
            initial_guess, 
            method='L-BFGS-B', 
            bounds=joint_limits,
            tol=1e-8,
            options={'maxiter': 100}
        )

        if res.success:
            return res.x
        else:
            print(f"IK Warning: Optimization failed. Reason: {res.message}")
            return res.x # 返回最后一次迭代的结果


class ErrorModelingUtils:
    #=====================================================
    # 机械臂误差建模方法
    #=====================================================
    @staticmethod
    def matrix_error(T_curr, T_target):
        """
        计算两个变换矩阵之间的位姿误差
        返回一个 6 维向量：[旋转误差(3), 平移误差(3)]
        """
        # 平移误差
        p_err = T_target[:3, 3] - T_curr[:3, 3]
        
        # 旋转误差 (使用 R_err = R_curr^T @ R_target 转换为轴角误差)
        R_err_mat = T_curr[:3, :3].T @ T_target[:3, :3]

        # 简单的迹计算得到旋转角度
        cos_theta = (np.trace(R_err_mat) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if abs(theta) < 1e-6:
            r_err = np.zeros(3)
        else:
            # 提取旋转轴
            expr = 1 / (2 * np.sin(theta))
            r_err = theta * expr * np.array([
                R_err_mat[2, 1] - R_err_mat[1, 2],
                R_err_mat[0, 2] - R_err_mat[2, 0],
                R_err_mat[1, 0] - R_err_mat[0, 1]
            ])
        
        return np.concatenate([r_err, p_err])
    
    @staticmethod
    def mdh_differential_error(a, alpha, d, theta, delta_params):
        """
        计算单关节的误差微分矩阵 Delta。
        该矩阵描述了由于几何参数微小偏差产生的位姿变动：T_real = T_nominal * (I + Delta)
        
        :param a, alpha, d, theta: 当前关节的标称 MDH 参数
        :param delta_params: 5维误差向量 [da, d_alpha, d_d, d_theta, d_beta]
        :return: 4x4 的微分变换矩阵 Delta
        """
        da, d_alpha, dd, d_theta, d_beta = delta_params
        
        # 预计算三角函数（标称值）
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        # 基于微分变换原理构造 Delta 矩阵
        # Delta = [  0, -dz,  dy, dx ]
        #         [  dz,  0, -dx, dy ]
        #         [ -dy, dx,   0, dz ]
        #         [  0,   0,   0,  0 ]
        # 但在参数辨识中，我们通常直接构造 Delta = Σ (∂T/∂pi * dpi) 的算子形式
        
        # 这里采用 Hayati-Mirmirani 改进模型中常见的微分算子结构：
        # d_T = (∂T/∂a)da + (∂T/∂α)dα + (∂T/∂d)dd + (∂T/∂θ)dθ + (∂T/∂β)dβ
        
        # 构造算子 (以旋转误差和平移误差组合)
        # 注意：这里的实现对应的是变换到局部坐标系下的等效微分矩阵
        Delta = np.zeros((4, 4))
        
        # 1. 角度误差产生的旋转部分 (Sk-symmetric)
        # d_theta 绕 Z 轴，d_alpha 绕 X 轴，d_beta 绕 Y 轴
        omega_z = d_theta + d_beta * sa  # 综合旋转分量
        omega_y = d_beta * ca
        omega_x = d_alpha
        
        Delta[0, 1] = -omega_z
        Delta[0, 2] =  omega_y
        Delta[1, 0] =  omega_z
        Delta[1, 2] = -omega_x
        Delta[2, 0] = -omega_y
        Delta[2, 1] =  omega_x
        
        # 2. 长度误差产生的平移部分
        # dx = da, dy = d_theta*a (位移近似), dz = dd
        # 注意：具体的项取决于你参考的论文对坐标系的定义，以下为通用 MDH 微分模型
        Delta[0, 3] = da
        Delta[1, 3] = a * d_theta + sa * dd # 耦合项
        Delta[2, 3] = ca * dd
        
        return Delta
