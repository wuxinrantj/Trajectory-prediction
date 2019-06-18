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

n_dots = 2200
T = 1

X = np.linspace(0, 11, n_dots)
sine = 5 * np.sin(2 * np.pi * X / T)
ex = np.exp(-0.2 * X) #+ np.random.rand(n_dots)
X1 = np.linspace(0, 11, 100)
res = np.random.rand(100) - 0.5
Y = sine * ex
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

'''
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

X2 = X [-600:]
Y2 = Y [-600:]
degrees = [2, 3, 5, 10]
results = []
for d in degrees:
    model = polynomial_model(degree=d)
    model.fit(X2, Y2)
    train_score = model.score(X2, Y2)  #训练集上拟合的怎么样
    mae = mean_absolute_error(Y2, model.predict(X2))  #均方误差 cost
    results.append({"model": model, "degree": d, "score": train_score, "mae": mae})
for r in results:
    print("degree: {}; train score: {}; mean absolute error: {}".format(r["degree"], r["score"], r["mae"]))

plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
for i, r in enumerate(results):
    fig = plt.subplot(2, 2, i+1)
    plt.title("Degree={}; MAE: {:f}".format(r["degree"], r["mae"]))
    #plt.scatter(X, Y, s=5, c='b', alpha=0.5)
    plt.plot(X, Y, 'b-')
    plt.plot(X2[-200:], r["model"].predict(X2)[-200:], 'r-')
plt.show()
