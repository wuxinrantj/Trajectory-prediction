# -*- coding: utf-8 -*-
# this is the module of my functions

import pandas as pd
from pandas import Series,DataFrame
from numpy import *
from hmmlearn import hmm
from geopy.distance import great_circle

# 数据预处理部分 =========================================================================================================================

def train_set_to_df(train_set):
    '''
    该函数将刚刚加载的源数据转换为对应的DateFrame数据
    '''
    df = DataFrame({'id':train_set[:,0],'time':train_set[:,1],'lon':train_set[:,2],'lat':train_set[:,3]})
    return df



### 经纬度到坐标的换算公式----------------------------------------------------------------------------------------------------------------

# 通过计算可以得到在北京地区经纬度到米的换算公式（近似计算）
# 1lon = 111km * sin(40度) = 82700m
# 1lat = 111000m
# x从经度的116度开始
# y从纬度的39度开始

def lon_to_x(lon):
    return (lon-116)*82700

def lat_to_y(lat):
    return (lat-39)*111000

### 坐标到经纬度的换算公式
# （虽然不一定会用到）

def x_to_lon(x):
    return (x/82700)+116

def y_to_lat(y):
    return (y/111000)+39


# 极坐标转换-------------------------------------------------------------------------------------------------------------------------------
# x，y 转角度 --------------------

# 下列函数为tan到角度的转换，在以下计算中默认使用角度，即第四个函数，其他函数可能会用到，先写下来
# 为什么使用角度而不使用弧度？弧度自然方便计算，但这里不需要算弧，而角度的感性认知度强，所以选角度

# todo : 这里将角度归到[0,360)有待商榷，因为10度和350度区别不大，但数字上区别很大。先跑一个demo，优化的时候可以考虑归到[-180,180)
# diata 已经在df数据归一化中归到了 [-180,180) 不要重复操作

def tan_to_rad(x,y):
    '''函数：输入x，y，返回反正切的弧度值，并符合极坐标中的范围[0,360)'''
    tmp = math.atan(y/x)
    if y>=0:
        if tmp>0:   # 第一象限
            pass
        else:       # 第二象限
            tmp = pi+tmp
    else:
        if tmp>0:   # 第三象限
            tmp = pi+tmp
        else:       # 第四象限
            tmp = 2*pi+tmp
    return tmp

def rad_without_pi(rad):
    '''函数：输入弧度，提出pi'''
    return rad/pi

def rad_to_deg(rad):
    '''函数：输入弧度，返回度'''
    return rad*180/pi

def deg_to_rad(deg):
    '''函数：输入度，返回弧度'''
    return deg*pi/180

def tan_to_deg(x,y):
    '''函数：输入x，y，返回反正切的度，并符合极坐标中的范围[0,2*pi)'''
    return rad_to_deg(tan_to_rad(x,y))


def xy_to_len(x,y):
    '''x，y 转长度L'''
    return sqrt(x**2+y**2)


# 向量化
vec_tan_to_rad = vectorize(tan_to_rad)
vec_rad_without_pi = vectorize(rad_without_pi)
vec_rad_to_deg = vectorize(rad_to_deg)
vec_tan_to_rad = vectorize(tan_to_deg)
vec_xy_to_len = vectorize(xy_to_len)



# 输入车的 id 返回stops ------------------------------------------------------------------------------------------------------------------
# (其实我觉得叫restart更好，但是stops前面一直在用，且比较顺口，所以就没改，优化、重构代码的时候可以考虑改为restart)

def find_stops(car_id,df):
    '''
    返回超过阈值的离散值
    返回一维list，状如[784,926,1332,1520]
    '''
    stop_time = 40         # 此参可调！
    stops = []
    tmp = df[df['id']==car_id]
    stop = 0
    for i in range(len(tmp)):
        if pd.isnull(tmp.iloc[i].theta):
            stop += 1
        else :
            if stop>stop_time:
                stops.append(i)
            stop=0
    return stops           # 返回一维list，状如[784,926,1332,1520]


def find_thepoint_stops(car_id,df):
    '''
    返回每一点的等待值(连续值)
    返回一维list，状如[0,0,0,0,0,0,5,0,0,0,2,0,0,0,...,0,0,23,0,0,...,0,0,0,52,0,0,1,...]共1600个数
    '''
    stops = []
    tmp = df[df['id']==car_id]
    stop = 0
    for i in range(len(tmp)):
        if pd.isnull(tmp.iloc[i].theta):
            stop += 1
        else :
                stop=0
        stops.append(stop)
    return stops            # 返回一维list，状如[0,0,0,0,0,0,5,0,0,0,2,0,0,0,...,0,0,23,0,0,...,0,0,0,52,0,0,1,...]共1600个数

# 其实这里是可以向量化一下的，但是为了代码的可读性和易修改性，我还是决定以外部循环的方式进行批量操作
# todo ：据说向量化函数比外部循环的速度要快，这点我还没有实验验证
#        如果确实有必要，重构的时候可以考虑使用向量化函数




# 模型训练部分====================================================================================================================
# 预测函数
# 传入状态(该函数已弃用)
def last_state_to_predict_next_position(last_state,last_point):
    '''
    该函数已弃用
    这里接受hmm.predict()解码后的状态和上一点的cdf行
    用状态对应的v和上一点的theta进行计算
    返回得出的下一点的x,y值
    '''
    v = state_v[last_state]
    theta_rad = deg_to_rad(last_point.theta)
    vx = v*cos(theta_rad)
    vy = v*sin(theta_rad)
    return last_point.x+vx,last_point.y+vy

# 传入预测的观测值的期望
def next_v_to_predict_next_position(x,y,v,theta,u=0,diata=0):
    '''
    这里接受hmm.predict()解码后的状态和上一点的cdf行
    用状态对应的v和上一点的theta进行计算
    返回得出的下一点的x,y值
    '''
    theta_rad = deg_to_rad(theta)
    diata_rad = deg_to_rad(diata)
    vx = v*cos(theta_rad)
    vy = v*sin(theta_rad)
    ux = u*cos(diata_rad)
    uy = u*sin(diata_rad)
    return x+vx+ux,y+vy+uy

# 向量化
vec_next_v_to_predict_next_position = vectorize(next_v_to_predict_next_position)

def distance(true_x,true_y,predict_x,predict_y):
    return great_circle((true_x,true_y),(predict_x,predict_y)).m

# 向量化
vec_distance = vectorize(distance)


# 模型效果展示==================================================================================================================================
def prediction(model,train_df,test_df,traincol_name,base_point=1000,anwser_point=1001):
    '''
    注意：
    模型和传入列是对应的，
    比如，用'v'训练出的model，使用该函数时，traincol_name只能传入['v']
       用'v','u'训练出的model，使用该函数时，traincol_name只能传入['v','u']
    
    传入参数为：
    model object 选定的模型
    DataFrame 用于训练的数据(这里并没有训练，只是找出state对应的特征值)
    DataFrame 传入由于测试的数据
    list    指定用于训练和预测的列
    int     传入观测点(基准点)在每个路径中的编号，缺省值为1000
    int     传入答案点在每个路径中的编号，缺省值为1001
    '''

    # 选择模型model
    model = model

    # 指定用于训练和预测的列
    traincol = array(test_df[traincol_name])
    traincol2 = array(train_df[traincol_name])  # 注意，这里代码命名不规范，对train_df进行解码仅用于储存对应的obs
    
    Z = model.decode(traincol)
    test_df['label'] = DataFrame(Z[1]).set_index(test_df.index)
    Z2 = model.decode(traincol2)
    train_df['label'] = DataFrame(Z2[1]).set_index(train_df.index)

    state_num = len(model.startprob_)
    
    print '---------*观察报告*---------'
    print '每种状态对应的轨迹数'
    for i in range(state_num):
        tmp = test_df[test_df['label']==i]
        print '状态',i,'有',len(tmp),'条轨迹'

    # 从均值的角度观察一下数据，发现确实是有效果的，并将状态对应的v储存
    tmp = 0
    state_obs=[]
    print '每种状态对应的观测'
    for tmp in range(state_num):
        tmp1 = train_df[train_df['label']==tmp]
#         state_obs.append([mean(tmp1.v),mean(tmp1.u),mean(tmp1.diata),mean(tmp1.stops_time)])
        state_obs.append([mean(tmp1.v)])


    # 每条轨迹长度
    min_id = min(test_df['id'])
    length = len(test_df[test_df['id']==min_id])
    # 轨迹数
    car_num = int(max(array([test_df['id']]).flatten())-min(array([test_df['id']]).flatten())+1)
    print '每条轨迹长度:',length
    print '轨迹数:',car_num

    # 设置基准点和答案点
    base_points = test_df.iloc[[i*length+base_point for i in range(car_num)]]                     
    anwser_points = test_df.iloc[[i*length+anwser_point for i in range(car_num)]]

    base_points_v = array(base_points[traincol_name])

    state_obs_col_vec = array(state_obs)
    where_are_nan = isnan(state_obs_col_vec)  
    state_obs_col_vec[where_are_nan] = 0 
    print '观测状态矩阵'
    
    if 'v' not in traincol_name:state_obs_col_vec[:,0]=0
    if 'u' not in traincol_name:state_obs_col_vec[:,1]=0
    if 'diata' not in traincol_name:state_obs_col_vec[:,2]=0
    if 'stops_time' not in traincol_name:state_obs_col_vec[:,3]=0

    print state_obs_col_vec

    # 上一点状态的概率矩阵
    last_states_proba = model.predict_proba(base_points_v) #这里没有设置length，但是有可能需要，否则有可能被当做是同一个序列
    # 下一点状态的概率矩阵
    predict_proba_next_points = dot(last_states_proba,model.transmat_)

    # 通过下一点状态的概率矩阵进行对应速度均值，得出的预测速度
    predict_next_points_obs = dot(predict_proba_next_points,state_obs_col_vec)
    print '观察前5个下一点的观测值'
    print predict_next_points_obs[:5]
    
    predict_next_position = []
    tmp=0
    for base_point in base_points.itertuples():
        tmp1 = predict_next_points_obs[tmp]
        predict_next_position.append(next_v_to_predict_next_position(base_point,v=tmp1[0],u=tmp1[1],diata=tmp1[2]))
        tmp+=1

    predict_next_position = array(predict_next_position)
    print '取前5个预测点观察'
    print predict_next_position[:5]
    errs = vec_distance(array(anwser_points.x),array(anwser_points.y),predict_next_position[:,0],predict_next_position[:,1])
    print '欧氏距离误差均值',mean(errs)
    print '欧氏距离误差标准差',std(errs)
    return predict_next_points_obs
    
    
    
    
# 每种状态对应的v均值,返回每个状态对应的v的表
def state_mean_v(model,train_df):
    '''
    返回一个array，顺序和model中的各项顺序一致
    
    这样使用：
    state_to_v = state_mean_v(model,train_df)
    v = state_to_v[state]
    '''
    # 指定用于训练和预测的列
    train_v = array(train_df['v']).reshape(-1,1)
    Z = model.decode(train_v)
    train_df['label'] = DataFrame(Z[1]).set_index(train_df.index)

    state_num = len(model.startprob_)
    state_to_v = []
    for tmp in range(state_num):
        tmp1 = train_df[train_df['label']==tmp]
        state_to_v.append(mean(tmp1.v))
    
    state_to_v = array(state_to_v)
    where_are_nan = isnan(state_to_v)  
    state_to_v[where_are_nan] = 0 
    
    return state_to_v
    
    
    
# 仅使用v的prediction==================================================================================================================================
def prediction_v(model,test_df,state_to_v):
    '''
    注意：
    test_df 是上一点的v（注意，仅有这一个时刻）
    返回下一点的nv_df,即next v
    state_to_v是状态到v的对应关系array
    '''
    test_v = array(test_df['v']).reshape(-1,1)
    Z = model.decode(test_v)
    test_df['label'] = DataFrame(Z[1]).set_index(test_df.index)

    state_num = len(model.startprob_)
    # 每条轨迹长度
    min_id = min(test_df['id'])
    length = len(test_df[test_df['id']==min_id])
    # 轨迹数
    car_num = int(max(array([test_df['id']]).flatten())-min(array([test_df['id']]).flatten())+1)

    # 上一点状态的概率矩阵
    last_states_proba = model.predict_proba(test_v) #这里没有设置length，但是有可能需要，否则有可能被当做是同一个序列
    # 下一点状态的概率矩阵
    predict_proba_next_points = dot(last_states_proba,model.transmat_)

    # 通过下一点状态的概率矩阵进行对应速度均值，得出的预测速度
    predict_next_points_v = dot(predict_proba_next_points,state_to_v)

    return predict_next_points_v
    
    
# 迭代函数
def predict_next_n_points(last_point_df,n,model,state_to_v):
    """
    x,y,v,theta 原始点参数
    n 为int 预测步数
    model 为使用的模型
    state_to_v 为state与v的对应关系
    返回prediction_next_n_points ，为(x,y)
    """
    x = array(last_point_df.x)
    y = array(last_point_df.y)
    v = array(last_point_df.v)
    theta = array(last_point_df.theta)
    next_n_points_location = []
    for step in range(n):
        test_df = DataFrame()
        test_df['id'] = last_point_df['id']
        test_df['x'] = last_point_df.x
        test_df['y'] = last_point_df.y
        test_df['v'] = last_point_df.v
        v = prediction_v(model,test_df,state_to_v)
        next_position = vec_next_v_to_predict_next_position(x,y,v,theta)
        next_n_points_location.append(next_position)
        x = next_position[0]
        y = next_position[1]
           
    return array(next_n_points_location)


# 封装函数，返回欧氏距离误差
def over_error(model,cdf,ctdf,last_point_num=279,steps=20):
    """
    最终的封装函数，返回欧氏距离误差
    注意：这里是仅使用给定点进行多步预测
    model 是使用的模型
    last_point_num=279 是last point
    steps=20 是预测步数
    ctdf=ctdf 是测试集
    cdf=cdf 是训练集
    
    """
    tmp = ctdf[ctdf.index%2100==last_point_num]
    state_to_v = state_mean_v(model,cdf)
    next_n_points_location = predict_next_n_points(tmp,steps,model,state_to_v)
    X = next_n_points_location[:,0,:].flatten()
    Y = next_n_points_location[:,1,:].flatten()
    tmpX = array(ctdf[ctdf.index%2100 == last_point_num].x)
    for i in range(1,steps):
        tmpX = append(tmpX,array(ctdf[ctdf.index%2100 == last_point_num+i].x))
    tmpY = array(ctdf[ctdf.index%2100 == last_point_num].y)
    for i in range(1,steps):
        tmpY = append(tmpY,array(ctdf[ctdf.index%2100 == last_point_num+i].y))
    
    error = mean(vec_distance(tmpX,tmpY,X,Y))
    return error



def predict_next_n_points1(last_point_df,n,model,state_to_v):
    """
    仅配合over_error1使用
    x,y,v,theta 原始点参数
    n 为int 预测步数
    model 为使用的模型
    state_to_v 为state与v的对应关系
    返回prediction_next_n_points ，为(x,y)
    """
    x = array(last_point_df.x)
    y = array(last_point_df.y)
    v = array(last_point_df.v)
    theta = array(last_point_df.theta)
    next_n_points_location = []
    for step in range(n):
        test_df = DataFrame()
        test_df['id'] = last_point_df['id']
        test_df['x'] = DataFrame(x).set_index(test_df.index)
        test_df['y'] = DataFrame(y).set_index(test_df.index)
        test_df['v'] = DataFrame(v).set_index(test_df.index)
        v = prediction_v(model,test_df,state_to_v)
        next_position = vec_next_v_to_predict_next_position(x,y,v,theta)
        next_n_points_location.append(next_position)
        x = next_position[0]
        y = next_position[1]
           
    return array(next_n_points_location)

def over_error1(model,cdf,ctdf,last_point_num=279,steps=20):
    """
    注意：这个函数是真·隐马模型，
    即每步预测是依据上一次预测点进行预测的，但效果没一次性预测好
    最终的封装函数，返回欧氏距离误差
    model 是使用的模型
    last_point_num=279 是last point
    steps=20 是预测步数
    ctdf=ctdf 是测试集
    cdf=cdf 是训练集
    
    """
    tmp = ctdf[ctdf.index%2100==last_point_num]
    state_to_v = state_mean_v(model,cdf)
    next_n_points_location = predict_next_n_points1(tmp,steps,model,state_to_v)
    X = next_n_points_location[:,0,:].flatten()
    Y = next_n_points_location[:,1,:].flatten()
    tmpX = array(ctdf[ctdf.index%2100 == last_point_num].x)
    for i in range(1,steps):
        tmpX = append(tmpX,array(ctdf[ctdf.index%2100 == last_point_num+i].x))
    tmpY = array(ctdf[ctdf.index%2100 == last_point_num].y)
    for i in range(1,steps):
        tmpY = append(tmpY,array(ctdf[ctdf.index%2100 == last_point_num+i].y))
    
    error = mean(vec_distance(tmpX,tmpY,X,Y))
    return error



# 封装函数，返回欧氏距离误差
# 注意：这个的测试集长度为200
def over_error_200(model,cdf,ctdf,last_point_num=279,steps=20,r=200):
    """
    最终的封装函数，返回欧氏距离误差
    注意：这里是仅使用给定点进行多步预测
    model 是使用的模型
    last_point_num=279 是last point
    steps=20 是预测步数
    ctdf=ctdf 是测试集
    cdf=cdf 是训练集
    
    """
    tmp = ctdf[ctdf.index%200==last_point_num]
    state_to_v = state_mean_v(model,cdf)
    next_n_points_location = predict_next_n_points(tmp,steps,model,state_to_v)
    X = next_n_points_location[:,0,:].flatten()
    Y = next_n_points_location[:,1,:].flatten()
    tmpX = array(ctdf[ctdf.index%200 == last_point_num].x)
    for i in range(1,steps):
        tmpX = append(tmpX,array(ctdf[ctdf.index%200 == last_point_num+i].x))
    tmpY = array(ctdf[ctdf.index%200 == last_point_num].y)
    for i in range(1,steps):
        tmpY = append(tmpY,array(ctdf[ctdf.index%200 == last_point_num+i].y))
    
    point_error = vec_distance(tmpX,tmpY,X,Y)
    error = mean(point_error)
    acc = len(point_error[point_error<=r]) * 1.0 / len(point_error)
    return error,acc