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

#  mnist 是灰度手寫數字圖片資料庫
#  batch_size是整數，指定进行梯度下降时每个batch包含的样本数
batch_size = 128
#  10個類別標籤  0-9
num_classes = 10
epochs = 1

"""
將圖片預處理
"""
#  圖片28*28的尺寸
img_rows, img_cols=28, 28
#  下載資料
#  X shape (60,000 28x28), y shape (10,000, )
(x_train, y_train), (x_test, y_test)= mnist.load_data()
#  reshape：将tensor变换为参数shape的形式，shape是列表形式
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#  28*28大小的黑白圖片
input_shape = (img_rows, img_cols, 1)
#  astype 轉換數據類型 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#  標準化，從0-255變成0-1之間
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#  將0-9之間的y改造成一個大小為10的向量，屬於哪個數字，
#  就在哪個位置為1，其他為0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
建立神經網路
"""
#  建立神經層
model = Sequential()  
#  32個3*3filters 、relu 激活函數、28*28*1的輸入，圖片28*28尺寸,1表黑白
#  Conv2D，2D空間卷積層
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#  64個3*3filters、relu 激活函數、input_shape為Conv2D在第一層使用時才用
model.add(Conv2D(64, (3, 3), activation='relu'))
#  用2*2的最大池化矩陣去壓縮資料
model.add(MaxPooling2D(pool_size=(2, 2)))
#  Dropout在训练中每次更新时， 
#  可隨機刪除部分神經元，防止過擬合
model.add(Dropout(0.25))
#  Flatten，建立平坦層，將64個12*12的矩陣，轉換成一維向量9216個神經元
model.add(Flatten())
#  Dense，神經元連接層，128個神經元，激活函數：relu
model.add(Dense(128, activation='relu'))
#  刪除部分神經元
model.add(Dropout(0.5))
#  Dense，輸出層，10個神經元，輸出維度10的矩陣，激活函數：softmax
model.add(Dense(num_classes, activation='softmax'))
#  compile編譯模型以提供訓練
#  loss：損失函數
#  optimizer：優化器
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])
#  fit 訓練模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
#  返回一个测试误差的标量值，或一个标量的list
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
