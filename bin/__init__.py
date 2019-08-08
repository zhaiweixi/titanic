import pymysql

import gensim

import redis

from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import sklearn.svm
import sklearn.ensemble

from sklearn.preprocessing import OneHotEncoder

import math

from gensim.models import word2vec

import redis

from rediscluster import StrictRedisCluster


import re
pattern = re.compile(r'(?<![\.\d])(?:\d{1,3}\.){3}\d{1,3}(?![\.\d])')
pattern2 = re.compile(r'[\u4E00-\u9FA5]+')
p2 = re.compile(r'[a-zA-Z0-9]+|[\u4E00-\u9FA5]+')
st = u"hello,world!!%[545]你好 世界。。。"
ste = re.sub("^[A-Za-z0-9]+", "", st)

new_s = ""
for s in st:
    if p2.match(s):
        new_s += s

print(new_s)