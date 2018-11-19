import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from pandas.core.frame import DataFrame
import csv
import time


def workday(date):
    # 是否是工作日，是则返回True，按照2018-02-12 17:40:51格式解析时间
    day = 0
    h = 0
    week = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    # print(week[6])
    if week[6]<5:
        day = 1
    if week[3]>7 and week[3]<22:
        h = 1
    h = week[3]
    m = week[4]
    return week[6], h, m


def add_time(path):
    # 按照出发时间，添加上weekday信息，0代表周一，6代表周日
    data = pd.read_csv(path, header=0)
    print('数据规模', data.shape)
    x = int(data.shape[0])
    H = []
    M = []
    someset = data.iloc[0:x, 1:]
    for i in range(x):
        date = someset.loc[i, 'start_time']
        week, h, m = workday(date)
        # print(week)
        H.append(h)
        M.append(m)
    H = DataFrame(H)
    someset['hour'] = H
    M = DataFrame(M)
    someset['min'] = M
    # print(someset.head(5))
    someset.to_csv(path)


def add_workday(path):
    # 按照出发时间，添加上time信息
    data = pd.read_csv(path, header=0)
    x = int(data.shape[0])
    w = []
    someset = data.iloc[0:x, 1:]
    for i in range(x):
        date = someset.loc[i, 'start_time']
        week,h,m = workday(date)
        # print(week)
        w.append(week)
    w = DataFrame(w)
    someset['week'] = w
    # print(someset.head(5))
    someset.to_csv(path)


def cut(name, percentage):
    # 剪切出一部分数据
    data = pd.read_csv(name, header=0)
    print('数据规模', data.shape)
    x = int(data.shape[0] * percentage)
    someset = data.iloc[0:x, :]
    someset.to_csv('./data/train_1.csv')



# name = './data/train_new.csv'
path = './data/train_1.csv'
data = pd.read_csv(path, header=0)
# 粗略查看
# print(type(data))
# print(data.head(10))
add_time(path)
# pre-process
# print(data.loc[1:5,'start_time':'end_time'])
# time = data.loc[0:data.shape[0]-1,'start_time':'end_time']

# 初步用贝叶斯进行train的部分集进行训练