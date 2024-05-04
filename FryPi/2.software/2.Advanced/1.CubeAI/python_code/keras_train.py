# 导入工具包
import tensorflow as tf
import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('2.software//2.Advanced//1.CubeAI//python_code//sin_values.csv', sep=',', header=None)
raw_x = data.iloc[:,0].astype(float)  # 提取输入数据
sinex = data.iloc[:,1].astype(float)  # 提取标签数据
print(sinex.shape)  # 打印标签数据的形状

# 建立模型
model = tf.keras.Sequential()  # 创建序贯模型
model.add(tf.keras.layers.Dense(units=10, activation='tanh', input_shape=(1,)))  # 添加全连接层，10个神经元，tanh激活函数，输入形状为1
model.add(tf.keras.layers.Dense(units=5, activation='tanh'))  # 添加全连接层，5个神经元，tanh激活函数
model.add(tf.keras.layers.Dense(units=1))  # 添加全连接层，1个神经元
model.summary()  # 打印模型结构

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),  # 编译模型，使用Adam优化器，学习率为0.001
            loss=tf.keras.losses.mse,                      # 损失函数为均方差
            metrics=[tf.keras.metrics.mse])  # 评估指标为均方差
history = model.fit(x=raw_x, y=sinex, epochs=2000)  # 拟合模型，训练2000个epoch

print(model.evaluate(raw_x, sinex))  # 打印模型在输入数据和标签数据上的评估结果

# 保存模型
model.save('./Embedded_things/sine_calcu.h5')

# 转换模型为tf lite格式 不量化
load_model = tf.keras.models.load_model('./Embedded_things/sine_calcu.h5')  # 加载模型
converter = tf.lite.TFLiteConverter.from_keras_model(load_model)  # 创建tflite转换器
tflite_model = converter.convert()  # 转换模型为tflite格式
open("./Embedded_things/sine_calcu.tflite", "wb").write(tflite_model)  # 将tflite模型写入文件