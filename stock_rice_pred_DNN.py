#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# build a simple rnn with cleaned test data to predict stock rice 
# 将股票涨幅看作分类问题

# step1: import modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import matplotlib.pyplot as plt
import os
import math

path = './test_data/'
stock_code = '600000'
filename = 'cleaned'+stock_code+'.csv'
df = pd.read_csv(path+filename)


# In[2]:


def rise_label(r):
    if r > 0.06:
        return 3
    elif r > 0.03:
        return 2
    elif r > 0:
        return 1
    else:
        return 0
df['rise_label'] = df['rise'].apply(lambda x: rise_label(x))


# In[3]:


# step2: 制定训练集和测试集
# 总数据集大小
data_sz = df.shape[0]
# 计算测试集大小, 约为原数据的%20. 向上取整
test_set_sz = math.ceil(data_sz * 0.2)
training_set_sz = data_sz - test_set_sz
# 前(data_sz - test_set_sz)天的涨幅作为训练集, 后test_set_sz天的涨幅作为测试集
training_set = df.iloc[0:training_set_sz, 2:3].values
training_label = df.iloc[0:training_set_sz, 3:4].values
test_set = df.iloc[-test_set_sz:, 2:3].values
test_label = df.iloc[-test_set_sz:, 3:4].values


# In[4]:


x_train = []
y_train = []

x_test = []
y_test = []
sample_sz = 15


# In[5]:


for i in range(sample_sz, training_set_sz):
    x_train.append(training_set[i - sample_sz:i, 0])
    y_train.append(training_label[i, 0])


# In[6]:


# 对训练集进行打乱
np.random.seed(56)
np.random.shuffle(x_train)
np.random.seed(56)
np.random.shuffle(y_train)
tf.random.set_seed(56)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)


# In[7]:


for i in range(sample_sz, test_set_sz):
    x_test.append(test_set[i - sample_sz:i, 0])
    y_test.append(test_label[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)


# In[8]:


class StockModel(Model):
    def __init__(self):
        super(StockModel, self).__init__()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(4, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        x = self.d1(x)
        y = self.d2(x)
        return y

model = StockModel()


# In[12]:



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/stock_rice_pred_DNN.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=8, epochs=50, validation_data=(x_test, y_test), validation_freq=1,
                   callbacks=[cp_callback])
model.summary()

file = open('./weights/stock_rice_pred_DNN_weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()


# In[10]:


# 预测
# 测试集输入模型进行预测
predicted_stock_rice = model.predict(x_test)
pred = tf.argmax(predicted_stock_rice, axis=1)
real_stock_rice_label = test_label[sample_sz:]
# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_rice_label, color='red', label='rice label')
plt.plot(pred, color='blue', label='Predicted rice label')
plt.title('stk_code '+stock_code + ' rice prediction')
plt.xlabel('Time')
plt.ylabel('Stock rice')
plt.legend()
plt.show()


# In[11]:


model.predict(x_test)


# In[ ]:




