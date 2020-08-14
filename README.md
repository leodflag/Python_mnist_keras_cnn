# Keras_CNN
This is a project about digital picture classification. (used Keras and TensorFlow)
## 目標
    用CNN訓練mnist手寫數據集
## 說明
    mnist為手寫數字0-9的黑白圖片數據集。
    圖片與對應標籤方面，數量為training data：55000筆，validation data：5000; 僅有圖片的test data：10000筆; 
    每張黑白圖片大小為[1,28,28,1]（張,寬,長,通道），有784個像素(pixels)，每個像素值都是數字0-255（0黑255白）。
    標籤與圖片的手寫數字對應，如圖片內為手寫數字5，標籤就設5，0-9種手寫圖片就有10個標籤，可用list儲存，如圖</br>片1,5,8,6,4對應標籤就是label=[1,5,8,6,4]
## 虛擬碼
    1. 下載mnist數據集，分成training data 跟test data
    2. 先處理圖片，將其轉換成列表形式，並透過標準化將所有像素值變成0-1之間
    3. 標籤處理成one hot encoding 形式
    4. 建立CNN網路
    5. 編譯模型
    6. 訓練模型
    7. 衡量模型訓練誤差
## CNN網路架構
![image](picture or gif url)
## 關閉GPU，使用CPU運算
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
## 使用GPU運算
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session=tf.InteractiveSession(config=config)

[連接文章](https://leodflagblog.wordpress.com/2020/08/14/%e4%bd%bf%e7%94%a8cnn%e5%b0%8dmnist%e6%89%8b%e5%af%ab%e6%95%b8%e6%93%9a%e9%9b%86%e9%80%b2%e8%a1%8c%e8%a8%93%e7%b7%b4%ef%bc%88cpu%e6%88%96gpu%e9%81%8b%e7%ae%97%ef%bc%89/)
