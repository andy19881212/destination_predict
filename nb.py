import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import csv
import time


def cut(name, percentage):
    data = pd.read_csv(name, header=0)
    print('数据规模', data.shape)
    x = int(data.shape[0] * percentage)
    someset = data.iloc[0:x, :]
    someset.to_csv('./data/train_1.csv')

def workday(day):
    # 是否是工作日，是则返回True，按照2018-02-12 17:40:51格式解析时间
    day = 0 # 1则是工作日
    time = 0 # 1则是工作时间
    week = time.strptime(day, "%Y-%m-%d %H:%M:%S")
    # print(week[6])
    if week[6]<5:
        day = 1
    if week[3]>7 and week[3]<22:
        time = 1
    return day,time


# name = './data/train_new.csv'
name = './data/train_1.csv'
data = pd.read_csv(name, header=0)
print(type(data))
print(data.head(10))

# pre-process
print(data.loc[1:5,'start_time':'end_time'])
time = data.loc[0:data.shape[0]-1,'start_time':'end_time']
print(type(data.loc[1,'start_time']))
holiday=[['1-1', '1-3'],
         ['10-1', '10-7']]
print(holiday)

# 初步用贝叶斯进行train的部分集进行训练