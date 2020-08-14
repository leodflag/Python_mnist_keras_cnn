# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 18:24:15 2018
@author: leodflag
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
"""
關閉GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

使用GPU
# for tf 1.15
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.InteractiveSession(config=config)
"""
#  mnist 是灰階手寫數字圖片資料庫
#  batch_size是整數，指定進行梯度下降時每個batch包含的樣本數
batch_size = 128
#  10個類別標籤  0-9
num_classes = 10
epochs = 1

"""
將圖片預處理
"""
#  圖片28*28的尺寸
img_rows, img_cols = 28, 28
#  下載資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#  X shape (60000,28,28), y shape (10000, )
#  reshape：將tensor變換成列表形式，樣本數量，幾列，幾行，通道數
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#  28*28大小的黑白圖片，通道為1
input_shape = (img_rows, img_cols, 1)
#  astype 轉換數據類型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#  標準化，將所有像素值從0-255變成0-1之間
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)  # (60000, 28, 28, 1)
print(x_train.shape[0], 'train samples')  # 60000
print(x_test.shape[0], 'test samples') # 10000
print(y_train.shape, 'y_train.shape')  # (60000,) 原本標籤數量
#  將標籤y  0-9只用0和1的陣列表示，有幾個類別就會有幾個數值，有10列，0類在[0]為1(其他為0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape, 'y_train.shape')  # (60000,10)  標籤數量,一個標籤內含10個資訊
"""
建立神經網路
"""
#  建立神經層
model = Sequential()
#  32個3*3 filters 、relu 活化函數、28*28*1的輸入
#  Conv2D，2D空間卷積層、input_shape為Conv2D在第一層使用時才用
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#  64個3*3 filters、relu 活化函數
model.add(Conv2D(64, (3, 3), activation='relu'))
#  用2*2的最大池化去壓縮資料
model.add(MaxPooling2D(pool_size=(2, 2)))
#  Dropout在訓練中每次更新時，將隨機位置對神經元乘上0
#  等於刪除隨機部分神經元，防止過擬合
model.add(Dropout(0.25))
#  Flatten，建立平坦層，將64個12*12的矩陣，轉換成一維向量9216個神經元
model.add(Flatten())
#  Dense，神經元連接層，128個神經元，活化函數：relu
model.add(Dense(128, activation='relu'))
#  刪除隨機部分神經元
model.add(Dropout(0.5))
#  Dense，輸出層，10個神經元，輸出維度10的矩陣，活化函數：softmax
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())
#  compile：編譯模型以提供訓練
model.compile(loss=keras.losses.categorical_crossentropy,   #  loss：損失函數
              optimizer=keras.optimizers.Adadelta(),  #  optimizer：優化器
              metrics=['accuracy']) # 評估標準
#  fit 訓練模型
model.fit(x_train, y_train,
          batch_size=batch_size,  # 一次迭代的訓練數量，因為數量太龐大會讓記憶體爆炸
          epochs=epochs,  # 迭代次數
          verbose=1, # 1輸出進度紀錄(預設)，0為不在標準輸出流輸出日誌訊息，2為每個epoch輸出一行紀錄
          validation_data=(x_test, y_test)) # 驗證集
#  衡量模型訓練誤差
score = model.evaluate(x_test, y_test, verbose=0) # 同fit的verbose，預設1，只能取0、1
print('Test loss:', score[0])  # 0.0514164837168064
print('Test accuracy:', score[1])  # 0.983299970626831
