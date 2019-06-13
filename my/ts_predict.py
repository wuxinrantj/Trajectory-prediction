#-*- encoding: utf-8 -*-

from __future__ import print_function

import os
import numpy as np
from argparse import ArgumentParser

from models import *
from get_data import get_train_data
from get_data import get_truth_data
from pct import *
from get_figure import *

def check_param(args):
    """检测命令行参数的合法性"""
    # 所有支持的模型
    model_list = ['lr', 'ann', 'lstm', 'arima']

    if args.model_name not in model_list:
        return 'unknown model'
    # 预测时间必须是整数，且不等于0
    if not isinstance(args.predict_time, int) and args.predict_time == 0:
        return 'error predict time'
    if not os.path.exists(args.data_dir):
        return 'the data file is not exist'
    else:
        return ''



if __name__ == "__main__":
    parser = ArgumentParser(description='Prediction of the time series.')

    parser.add_argument(
        '--model_name', default='lr',
        choices=names(), help='Name of the model to use.')

    parser.add_argument(
        '--data_dir', default='./data/data.csv',
        help='Dir of the data to train')

    parser.add_argument(
        '--predict_time', type=int,
        help='The prediction time.')
    args = parser.parse_args()

    check_result = check_param(args)

    if check_result == '':
        ori_data, timestamp_list, value_list = get_train_data(args.data_dir, args.predict_time)
        if len(value_list) < args.predict_time:
            print('less original data')
        else:
            create_model = create(args.model_name, predict_time=args.predict_time)
            train_model = create_model.train(ori_data, timestamp_list, value_list)
            if train_model is not None:
                predict_data = create_model.predict(train_model, value_list)
                print("the prediction result:")
                print(predict_data)

                truth_data = get_truth_data(args.data_dir, args.predict_time)
                if predict_data is not None and truth_data is not None:
                    pct_mean_value, RMSE = pct(predict_data, truth_data)
                    print("the prediction error:%f" % pct_mean_value)
                    print("the prediction RMSE:%f" % RMSE)
                    get_figure(value_list, predict_data, truth_data, RMSE)
                    print(len(value_list),len(predict_data),len(truth_data))
                else:
                    print('The result of prediction is null')
            else:
                print('The model is None')
    else:
        print(check_result)
