import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.io import loadmat
# from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from pandas.core.frame import DataFrame
from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import sklearn.metrics
import time
from math import radians, atan, tan, sin, acos, cos
import math
import os
import seaborn as sns

def get_score(d):
    # count the final score
    sum = 0
    for i in d:
        i = 1/(1 + math.exp(-(i - 1000)/250))
        sum = sum + i
    return sum/d.__len__()


def get_distance(latA, lonA, latB, lonB):
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)

    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
        return distance  # meter
    except:
        return 0.0000001


def analyze_workday(date):
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


def add_workday(path):
    # 按照出发时间，添加上time信息,添加上weekday信息，0代表周一，6代表周日
    data = pd.read_csv(path, header=0)
    x = int(data.shape[0])
    w = []
    H = []
    M = []
    someset = data.iloc[0:x, 1:]
    for i in range(x):
        date = someset.loc[i, 'start_time']
        week,h,m = analyze_workday(date)
        w.append(week)
        H.append(h)
        M.append(m)
    w = DataFrame(w)
    someset['week'] = w
    H = DataFrame(H)
    someset['hour'] = H
    M = DataFrame(M)
    someset['min'] = M
    # print(someset.head(5))
    someset.to_csv(path)


def cut(path):
    # cut dataset by id
    data = pd.read_csv(path, header=0)
    id = ''
    start = 0
    n = 0
    for i in range(data.shape[0]):
        if id == data.loc[i, 'out_id']:
            n += 1
        else:
            if id != '':
                new_set = data.iloc[start:start+n, 1:]
                new_set.to_csv('./data/' + str(id) + '.csv')
            id = data.loc[i, 'out_id']
            start = i
            n = 1


def draw_some_pic(n, path):
    # draw pic by n
    if n == 1:
        # draw data sets' size
        size = pd.read_csv(path, header=0).iloc[:,1:]
        print(size.shape)
        sns.distplot(size, color="b", bins=100)
        sns.distplot(size, hist=False)
        # sns.kdeplot(size, shade=True)
        plt.savefig('./pic/sets_size.jpg')
        plt.show()
    if n == 2:
        print('hehe')


def get_all_size(path):
    # path is csv files' directory, and it will count how much rows they have
    size = []
    list = os.listdir(path)
    for file in range(0, len(list)):
        filepath = os.path.join(path, list[file])
        if os.path.isfile(filepath):
            data = pd.read_csv(filepath, header=0)
            size.append(data.shape[0])
    size = pd.DataFrame(np.array(size), columns=['data_size'])
    size.to_csv('./data_size')


class NBclassifier():
    def __init__(self, clf_path=None, vec_path=None):
        """
        创建对象时完成的初始化工作，判断分类器与vector路径是否为空，
        若为空则创建新的分类器与vector，否则直接加载已经持久化的分类器与vector。
        """
        if (clf_path == None or vec_path == None):
            self.clf = GaussianNB()
            # self.vec = TfidfVectorizer()
        else:
            self.clf = joblib.load(clf_path)
            # self.vec = joblib.load(vec_path)
        # turn lat and lon from float into int
        self.times = 1000000

    def save_model(self, clf_path = './model/clf'):
        joblib.dump(self.clf, clf_path)

    def load_model(self, clf_path = './model/clf'):
        self.clf = joblib.load(clf_path)

    def NBaccura(self, table, pre):
        percent = sklearn.metrics.accuracy_score(table, pre, normalize=True)
        n = sklearn.metrics.accuracy_score(table, pre, normalize=False)
        print('percent is ',percent,' and num is ',n)
        return percent, n

    def reduce(self, table, pre, n = 100):
        table = (table / n).astype(int)
        pre = (pre / n).astype(int)
        return table, pre

    def enlarge(self, table, pre, n = 10000):
        # multiply table and prediction with n based on self.times , table is numpy.array
        table = (table * n).astype(int)
        pre = (pre * n).astype(int)
        return table, pre


path_1 = './data/4A23256745CBA3B0.csv'
path_2 = './data/2016061820000b.csv'

# path = './train_new.csv'
# data = pd.read_csv(path_2, header=0)
# print('data shape is :',data.shape)

# add_workday(path)

draw_some_pic(1, './pic/data_size')




a = 10/0
# 初步用贝叶斯进行train的部分集进行训练
nb = NBclassifier()
n_1 = 10000
n_2 = 10000

train_size = int(1/4 * data.shape[0])
test_size = data.shape[0] - train_size


x = data.loc[0:train_size,['week', 'hour', 'start_lat', 'start_lon']]
lat = data.loc[0:train_size,'end_lat']
x_pre = data.loc[train_size + 1:train_size + test_size, ['week', 'hour', 'start_lat', 'start_lon']]
table = data.loc[train_size + 1:train_size + test_size, 'end_lat']

# table must be int
table = np.array(table)
lat, table = nb.enlarge(lat, table, n=n_1)
print('lat is like', lat[0:2])
nb.clf.fit(x, lat)
pre = nb.clf.predict(x_pre)

table, pre = nb.reduce(table, pre, n=n_2)
print('pre is like', pre[0:2])
p, n = nb.NBaccura(table, pre)

# nb.save_model()

x = data.loc[128:700,['week', 'hour', 'start_lat', 'start_lon']]
lat = data.loc[128:700,'end_lat']
x_pre = data.loc[701:970,['week', 'hour', 'start_lat', 'start_lon']]
table = data.loc[701:970,'end_lat']

table = np.array(table)
lat, table = nb.enlarge(lat, table, n=n_1)
nb.clf.fit(x, lat)
pre = nb.clf.predict(x_pre)

table, pre = nb.reduce(table, pre, n=n_2)
p, n = nb.NBaccura(table, pre)

x = data.loc[971:1100,['week', 'hour', 'start_lat', 'start_lon']]
lat = data.loc[971:1100,'end_lat']
x_pre = data.loc[1101:1142,['week', 'hour', 'start_lat', 'start_lon']]
table = data.loc[1101:1142,'end_lat']

table = np.array(table)
lat, table = nb.enlarge(lat, table, n=n_1)
nb.clf.fit(x, lat)
pre = nb.clf.predict(x_pre)

table, pre = nb.reduce(table, pre, n=n_2)
p, n = nb.NBaccura(table, pre)