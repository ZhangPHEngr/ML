"""

"""
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
sys.path.append("../../")
from CommonUtilities.platform_info import *

print(check_platform("windows"))
# 生成全部数据平面
x1, y1 = np.mgrid[-2:2:200j, -2:2:200j]  # 生成二维扩展数组
z1 = x1 * np.exp(- x1 ** 2 - y1 ** 2)  # 函数

# 梯度下降法的数据
p = [0.5, 0.5]  # [x, y]
alpha = 0.05
x_p, y_p, z_p = [], [], []
for count in range(100):
    x = p[0]
    y = p[1]
    p[0] -= alpha*(np.exp(-x**2-y**2) - 2*x**2*np.exp(-x**2-y**2))
    p[1] -= -alpha*2*x*y*np.exp(-x ** 2 - y ** 2)
    x_p.append(p[0])
    y_p.append(p[1])
    z_p.append(x * np.exp(- x ** 2 - y ** 2))

# 绘图
fig = plt.figure()
ax = fig.gca(projection='3d')
# 绘制全局图像
ax.plot_surface(x1, y1, z1, alpha=0.8)  # 绘图，设置映射
# 绘制梯度下降图像
ax.plot(x_p, y_p, z_p, '-ro', label='Gradient Descent')
plt.show()




