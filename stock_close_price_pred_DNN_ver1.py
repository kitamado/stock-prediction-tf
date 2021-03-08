#!/usr/bin/env python
# coding: utf-8

# In[1]:


# build a simple rnn with cleaned test data to predict stock rice 
# 将股票涨幅看作二元分类问题
# 涨幅大于0记作1, 小于0记0
# step1: import modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import math

path = './test_data/'
stock_code = '600000'
filename = 'cleaned'+stock_code+'.csv'
df = pd.read_csv(path+filename)
df['rise_or_not'] = df['rise'].apply(lambda x: 1 if x > 0 else 0)

# In[3]:


# step2: 制定训练集和测试集
# 总数据集大小
data_sz = df.shape[0]
data_set = df.iloc[:, 2:3].values

# 数据归一化到(0,1)之间
sc = MinMaxScaler(feature_range=(0, 1))
# 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
data_set_scaled = sc.fit_transform(data_set) 

x_train = []
y_train = []

sample_sz = 15

# 利用for循环，遍历整个训练集，提取训练集中连续sample_sz=15天的开盘价作为输入特征x_train，
# 第16天的数据作为标签，for循环共构建training_set_sz-15组数据。
for i in range(sample_sz, len(data_set_scaled)):
    x_train.append(data_set_scaled[i - sample_sz:i, 0])
    y_train.append(data_set_scaled[i, 0])
# 对训练集进行打乱
np.random.seed(56)
np.random.shuffle(x_train)
np.random.seed(56)
np.random.shuffle(y_train)
tf.random.set_seed(56)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)


# In[4]:


model = tf.keras.models.Sequential(
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
)


# In[6]:


model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=200, validation_split=0.2, validation_freq=20)

model.summary()


# In[ ]:




