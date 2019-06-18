import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab  as plt
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import sys
from dateutil.relativedelta import relativedelta
from  copy import deepcopy
from statsmodels.tsa.arima_model import ARMA
import warnings
warnings.filterwarnings("ignore")

class arima_model:
    def __init__(self, ts, maxLag=9):
        self.data_ts = ts
        self.resid_ts = None
        self.predict_ts = None
        self.forecast_ts = None
        self.maxLag = maxLag
        self.p = maxLag
        self.q = maxLag
        self.properModel = None
        self.bic = sys.maxsize

    # 计算最优的ARIMA模型，将相关结果赋给相应的属性
    def get_proper_model(self):
        self._proper_model()
        self.predict_ts = deepcopy(self.properModel.predict())
        self.resid_ts = deepcopy(self.properModel.resid)
        self.forecast_ts = deepcopy(self.properModel.forecast())

    # 对于给定范围内的p,q计算拟合得最好的arima模型，这里是对差分好的数据进行拟合，故差分恒为0
    def _proper_model(self):
        for p in np.arange(self.maxLag):
            for q in np.arange(self.maxLag):
                model = ARMA(self.data_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1, method="css")
                except:
                    continue
                bic = results_ARMA.bic

                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.properModel = results_ARMA
                    self.bic = bic
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()

    # 参数确定模型
    def certain_model(self, p, q):
        model = ARMA(self.data_ts, order=(p, q))
        try:
            self.properModel = model.fit(disp=-1, method="css")
            self.p = p
            self.q = q
            self.bic = self.properModel.bic
            self.predict_ts = self.properModel.predict()
            self.resid_ts = deepcopy(self.properModel.resid)
            self.forecast_ts = self.properModel.forecast()
        except:
            print("You can not fit the model with this parameter p,q")

dateparse = lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')
#paese_dates指定日期在哪列 index_dates将年月日的哪个作为索引，date_parser将字符串转为日期
f = open("AirPassengers.csv")
data = pd.read_csv(f, parse_dates=["Month"], index_col="Month", date_parser=dateparse)
ts = data["#Passengers"]

def draw_ts(timeSeries, title):
    f = plt.figure(facecolor="white")
    timeSeries.plot(color="blue")
    plt.title(title)
    plt.show()


def seasonal_decompose(ts):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(ts, model="multiplicative")
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    draw_ts(ts, 'origin')
    draw_ts(trend, 'trend')
    draw_ts(seasonal, 'seasonal')
    draw_ts(residual, 'residual')
    return trend, seasonal, residual


def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    #     print ("dfoutput",dfoutput)
    return dfoutput


ts_log = np.log(ts)
trend,seasonal,residual = seasonal_decompose(ts_log)
seasonal_arr = seasonal
residual = residual.dropna()
residual_mean = np.mean(residual.values)
trend = trend.dropna()

#将原始数据分解为趋势分量，季节周期和随机分量
#对trend进行平稳定检验
testStationarity(trend)
#对序列进行平稳定处理
trend_diff_1 = trend.diff(1)
trend_diff_1 = trend_diff_1.dropna()
draw_ts(trend_diff_1,'trend_diff_1')
testStationarity(trend_diff_1)
trend_diff_2 = trend_diff_1.diff(1)
trend_diff_2 = trend_diff_2.dropna()
draw_ts(trend_diff_2,'trend_diff_2')
testStationarity(trend_diff_2)
# 使用模型拟合趋势分量
# 使用模型参数的自动识别
model = arima_model(trend_diff_2)
model.get_proper_model()
predict_ts = model.properModel.predict()

# 还原数据,因为使用的是乘法模型，将趋势分量还原之后需要乘以对应的季节周期分量和随机分量
diff_shift_ts = trend_diff_1.shift(1)
diff_recover_1 = predict_ts.add(diff_shift_ts)
rol_shift_ts = trend.shift(1)
diff_recover = diff_recover_1.add(rol_shift_ts)
recover = diff_recover['2153-1':'2169-9'] * seasonal_arr['2153-1':'2169-9'] * residual_mean
log_recover = np.exp(recover)
draw_ts(log_recover, 'log_recover')
#模型评价
ts_quantum = ts['2153-1':'2169-9']
plt.figure(facecolor = "white")
#
log_recover = log_recover / 10 - 5
ts_quantum = ts_quantum / 10 - 5

log_recover.plot(color = "blue",label = "Predict")
ts_quantum.plot(color = "red", label = "Original")
plt.legend(loc = "best")
plt.title("MAE %.4f" % (sum(abs(ts_quantum - log_recover)) / ts_quantum.size))
plt.show()