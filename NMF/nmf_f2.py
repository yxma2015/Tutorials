# -*-coding:utf-8 -*-

"""
# File       : nmf_f2.py
# Author     ：Yaxiong Ma
# E-Mail    : mayaxiong@xidian.edu.cn
# Create Time: 2024/5/21 9:56
# Description："2范数作为目标函数的NMF优化过程"
"""
import numpy as np
from matplotlib import pyplot as plt

def nmf_f2(X, k, max_iter=200, tol=1e-6):
    """
    使用2范数作为目标函数的非负矩阵分解（NMF）

    参数：
    X : 原始矩阵 (m x n)
    k : 低维度 (目标特征的个数)
    max_iter : 最大迭代次数
    tol : 收敛阈值

    返回：
    W : 低维表示矩阵 (m x k)
    H : 低维表示矩阵 (n x k)
    """
    m, n = X.shape
    W = np.random.rand(m, k)
    H = np.random.rand(n, k)
    loss = []
    for iteration in range(max_iter):
        epsilon = 1e-10

        # 更新 W
        W_numer = X@H
        W_denom = W@H.T@H + epsilon
        W *= W_numer / (W_denom)
        # 更新 H
        H_numer = X.T@W
        H_denom = H@W.T@W + epsilon
        H *= H_numer / (H_denom)


        # 计算F2范数
        WH = W@H.T
        f_2norm = np.linalg.norm(X-WH)**2

        if iteration == max_iter - 1:
            print(f"Iteration {iteration}: F2NORM = {f_2norm}")

        loss.append(f_2norm)
        # 收敛检查
        if f_2norm < tol:
            print(f"Converged at iteration {iteration} with F2 Norm = {f_2norm}")
            break
    plt.plot(loss)
    plt.title("the process of optimization of NMF",fontsize=20)
    plt.xlabel("iteration",fontsize=20)
    plt.ylabel("objective function", fontsize=20)
    plt.show()
    return W, H


# 示例数据

X = np.random.random((10,100))

# 指定低维度
k = 15

# 执行NMF
W, H = nmf_f2(X, k)

"""print("原始矩阵 X:")
print(X)
print("分解矩阵 W:")
print(W)
print("分解矩阵 H:")
print(H)"""

# 验证近似结果
X_approx = W@H.T
print("近似矩阵 X_approx:")
# print(X_approx)
print(X - X_approx)
