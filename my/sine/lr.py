from matplotlib.figure import SubplotParams
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import random

def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression(normalize=True)  #normalize=True对数据归一化处理
    pipeline = Pipeline([("polynomial_features", polynomial_features),#添加多项式特征
                         ("linear_regression", linear_regression)])
    return pipeline

n_dots = 3400
T = 1

X = np.linspace(0, n_dots/60, n_dots)
sine = 5 * np.sin(0.6 * np.pi * X / T)
ex = np.exp(-0.02 * X) #+ np.random.rand(n_dots)
X1 = np.linspace(0, 11, 100)
res = np.random.rand(100) - 0.5
Y = np.log(10 + sine * ex)
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
'''
decomposition = seasonal_decompose(Y, freq=200, model="multiplicative")
decomposition.plot()
plt.show()

fig = plt.subplot(4, 1, 1)
plt.plot(X, Y, 'r-')
fig = plt.subplot(4, 1, 3)
plt.plot(X, sine, 'r-')
fig = plt.subplot(4, 1, 2)
plt.plot(X, ex, 'r-')
fig = plt.subplot(4, 1, 4)
plt.plot(X1, res, 'r-')

plt.show()

'''

X2 = X[-600:]
Y2 = Y[-600:]
d = 20
results = []

model = polynomial_model(degree=d)
model.fit(X2, Y2)
train_score = model.score(X2, Y2)  #训练集上拟合的怎么样
mae = mean_absolute_error(Y2, model.predict(X2))  #均方误差 cost
results.append({"model": model, "degree": d, "score": train_score, "mae": mae})
for r in results:
    print("degree: {}; train score: {}; mean absolute error: {}".format(r["degree"], r["score"], r["mae"]))

plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))

fig = plt.plot()
plt.title("MAE: {:f}".format(r["mae"]))
#plt.scatter(X, Y, s=5, c='b', alpha=0.5)
Y = np.array(Y)
Y = Y * 4 - 8.7
Y_show = np.array(r["model"].predict(X2)[-200:])
Y_show = Y_show * 4 - 8.7
plt.plot(X, Y, 'b-')
plt.plot(X2[-200:], Y_show, 'r-')
plt.xlabel('Time(s)')
plt.ylabel('x_coordinate(m)')
plt.show()
