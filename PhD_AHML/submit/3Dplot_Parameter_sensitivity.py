# -*- coding:utf-8 -*-
# @Time : 2021/7/3 15:33
# @Author: LCHJ
# @File : 3dplot.py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('submit/Palm_Parameter_sensitivity.txt')  # 打开文件

# x = data[:, 0]  # 第2列，1-21行数据
# y = data[:, 1]  # 第2列，1-21行数据
# z = data[:, 2] # 第2列，1-21行数据
x = np.outer(np.linspace(0.00, 1.0, 11), np.ones(11))  # margin
y = np.outer(np.linspace(0.00, 0.012, 11), np.ones(11)).T  # margin
z = data
# x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
# y = x.copy().T  # transpose
# z = np.cos(x ** 2 + y ** 2)
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.figsize'] = (8, 7)
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.8,
                cmap='rainbow', edgecolor='none')
ax.contour(x, y, z, zdim='z', offset=50, cmap='rainbow')
ax.set_zlim(50, 100)
ax.set_title('Parameter sensitivity: ρ, γ')
ax.set_xlabel(' ρ', fontsize=18)
ax.set_ylabel(' γ', fontsize=18)
ax.set_zlabel('ACC ', fontsize=18)
plt.savefig("submit/Palm_Parameter_sensitivity.png")
plt.show()
