# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:10:10 2022

@author: nbrow
"""
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import numpy as np
def f(x, y):
    return np.sqrt((7-x)/(abs(y)))


n=np.linspace(1,7,100)
Y_Diff=np.linspace(-1,1,100)
Max_Steps=7
X, Y = np.meshgrid(n, Y_Diff)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(n, Y_Diff, Z,cmap='OrRd_r')
ax.view_init(0, -90)