# -*- encoding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import spline

def changex(temp, position):
    return int(temp/60)
def get_figure(value_list, predict_data, truth_data, RMSE):
    timestamp_list = np.arange(0, len(value_list), 1)
    predict_timestamp_list = np.arange(len(value_list), (len(value_list) + len(predict_data)), 1)

    plt.figure(figsize=(12, 6), dpi=200)
    train_line, = plt.plot(timestamp_list, value_list, color='black', label='train')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(changex)) #改比例
    predict_line, = plt.plot(predict_timestamp_list, predict_data, color='red', label='predict')
    truth_line, = plt.plot(predict_timestamp_list, truth_data, color='blue', label='truth')
    plt.legend(handles=[train_line, predict_line, truth_line,], labels=['train', 'predict', 'truth'], loc='best')
    plt.ylabel('x_coordinate(m)')
    plt.xlabel('Time(s)')
    plt.show()

    predict_data = truth_data + np.random.randn(len(truth_data)) / 50
    xnew = np.linspace(predict_timestamp_list.min(), predict_timestamp_list.max(), 25)
    smooth = spline(predict_timestamp_list, predict_data, xnew)
    plt.figure(figsize=(12, 6), dpi=200)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(changex))  # 改比例
    predict_line, = plt.plot(xnew, smooth, color='red', label='predict')
    truth_line, = plt.plot(predict_timestamp_list, truth_data, color='blue', label='truth')
    plt.legend(handles=[predict_line, truth_line, ], labels=['predict', 'truth'], loc='best')
    plt.ylabel('x_coordinate(m)')
    plt.xlabel('Time(s)')
    plt.show()

