import numpy as np
import matplotlib.pyplot as plt

dt = 0.001 #1stepの時間[sec]
times  =  np.arange(0,1,dt)
N = times.shape[0]

f  = 5  #サイン波の周波数[Hz]
sigma  = 0.5 #ノイズの分散

np.random.seed(1)
# サイン波
x_s =np.sin(2 * np.pi * times * f) 
x = x_s  +  sigma * np.random.randn(N)
# 矩形波
y_s =  np.zeros(times.shape[0])
y_s[:times.shape[0]//2] = 1
y = y_s  +  sigma * np.random.randn(N)

plt.plot(times,x,label="sin wave")
plt.plot(times,y,label="square wave")
plt.legend()
plt.show()
