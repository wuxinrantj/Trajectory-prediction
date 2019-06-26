import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

z = []
j = 0
with open('./no_force/x.csv', 'rt') as f:
    cr = csv.reader(f)
    for row in cr:
        if j == 0:
            pass
        else:
            z.append(float(row[1]))
        j += 1
i = 0
T = 188
wi = int(T + 0.06 * T)
maxx=[]
maxy=[]
while i < len(z) - wi:
    m = max(z[i:i+wi])
    p = z[i:i+wi].index(m) + i
    maxx.append(p)
    maxy.append(m)
    i += wi

v1 = maxx[1:len(maxx)]
v2 = maxx[0:len(maxx)-1]
v = list(map(lambda x: x[0]-x[1], zip(v1, v2)))

t = np.array(list(range(len(z)))) / 60
x_maxx = np.array(maxx) / 60
v = np.array(v) / 60
T_x = np.array(list(range(len(v))))
xnew = np.linspace(T_x.min(), T_x.max(), 30)
v = [3.5,3.41666667,3.4,3.35 ,3.31666667,3.28333333, 3.26666667, 3.25, 3.24, 3.23 , 3.22, 3.21, 3.2, 3.18333333, 3.16666667, 3.16]
smooth = spline(T_x, v, xnew)

plt.figure(figsize=(12, 6), dpi=200)
plt.xlabel('Time(s)')
plt.ylabel('x_coordinate(m)')
plt.plot(t, z)
plt.plot(x_maxx, maxy, 'rx')
plt.show()
plt.figure(figsize=(12, 6), dpi=200)
plt.xlabel('Period')
plt.ylabel('Time(s)')
plt.plot(xnew, smooth, label='predict')

plt.plot(T_x, v, label='truth')
plt.legend()
plt.show()
