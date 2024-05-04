import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        #只会输出错误信息
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime         #用于记录两个now时间，来计算一次训练的耗费时间

#------------------------------【加载数据】----------   

f = np.load("D:/Visual Studio Code/Open-source-projects/FryPi/2.software/2.Advanced/3.Thermal-camera-gesture-recognition/python_codes/camera_data/train_data.npz")
x_train, y_train = f['x_train'],f['y_train']
f.close()

print(x_train.shape, y_train.shape)

#------------------------------【查看数据】---------------------------------

image_index=230
print(y_train[image_index])     #查看随机一张图片的label
x_train = x_train.astype('float32')  #数据类型转换

# 计算最大值和最小值
max_value = np.max(x_train)
min_value = np.min(x_train)

# 最大最小值归一化操作
x_train_normalized = (x_train - min_value) / (max_value - min_value)

x_train = x_train_normalized.reshape(x_train.shape[0], 24, 32, 1)  #数据维度转换
plt.imshow(x_train[image_index])  #图片显示
plt.show()

# #------------------------------【搭模型】---------------------------------
# 使用Keras Sequential模型创建神经网络模型
model = tf.keras.models.Sequential([

    # 添加2D卷积层，使用5个大小为5x5的卷积核，采用valid padding，即不进行填充。，激活函数为ReLU，输入图像的大小为24x32，通道数为1（灰度图像）。
    tf.keras.layers.Conv2D(filters=5, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu, input_shape=(24, 32, 1)),
    
    # 添加2D最大池化层，池化窗口大小为2x2，使用same padding，保持特征图大小不变。
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    
    # 添加2D卷积层，使用3个大小为3x3的过滤器，采用valid填充，激活函数为ReLU，输入特征图的大小为10x14，通道数为5。
    tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding='valid', activation=tf.nn.relu, input_shape=(10, 14, 5)),
    
    # 与第一个池化层相同
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    
    # 展平层，用于将多维输入转换为一维输入
    tf.keras.layers.Flatten(),
    
    # 全连接层，输出维度为32，激活函数为ReLU
    tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
    
    # 全连接层，输出维度为16，激活函数为ReLU
    tf.keras.layers.Dense(units=16, activation=tf.nn.relu),
    
    # 输出层，输出维度为6，激活函数为softmax，用于多类别分类
    tf.keras.layers.Dense(units=6, activation=tf.nn.softmax)


    #卷积默认步幅为1，池化默认步幅为2，padding默认same，激活函数默认relu，全连接层默认激活函数默认linear
])

# 打印模型的摘要信息
model.summary()


# #------------------------------【训练】---------------------------------

# 超参数设置
# 定义训练参数
num_epochs = 50  # 训练轮数
batch_size = 64  # 批量大小
learning_rate = 0.001  # 学习率

# 优化器
# 创建Adam优化器对象，用于调整模型参数以最小化损失函数
adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

# 编译模型
# 对模型进行编译
model.compile(
    optimizer=adam_optimizer,  # 使用adam优化器
    loss=tf.keras.losses.sparse_categorical_crossentropy,  # 使用稀疏分类交叉熵损失函数
    metrics=['accuracy']  # 监控模型训练过程中的准确率
)
# 训练模型
# 获取当前时间
start_time = datetime.datetime.now()

# 使用模型拟合训练数据
# 使用训练数据x_train和标签y_train来训练模型
# 使用batch_size作为每次训练的批大小
# 将模型训练epochs次
model.fit(x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs)
# 获取结束时间
endtime = datetime.datetime.now()

# 计算时间消耗
time_cost = endtime - start_time
print('time_cost = ', time_cost)

# 保存/加载模型
model.save('2.software/2.Advanced/3.Thermal-camera-gesture-recognition/python_codes/gesture.h5')

# #------------------------------【评估】---------------------------------

# 加载测试数据文件
f = np.load("2.software/2.Advanced/3.Thermal-camera-gesture-recognition/python_codes/camera_data/test_data.npz")
# 从文件中获取测试数据和标签
x_test, y_test = f['x_test'],f['y_test']
# 关闭文件
f.close()
# 将测试数据转换为float32类型
x_test = x_test.astype('float32') #数据类型转换

# 计算最大值和最小值
# 计算 x_test 数组中的最大值和最小值
max_value = np.max(x_test)
min_value = np.min(x_test)

# 最大最小值归一化操作
# 对输入的测试数据进行归一化处理
x_test_normalized = (x_test - min_value) / (max_value - min_value)

# 将处理后的输入数据重塑为指定的形状
x_test = x_test_normalized.reshape(x_test.shape[0], 24, 32)
print(x_test.shape, y_test.shape)

# 加载训练好的模型
model = tf.keras.models.load_model('2.software/2.Advanced/3.Thermal-camera-gesture-recognition/python_codes/gesture.h5')

# 打印模型结构信息，得到的还是之前训练好的那个模型的结构信息
model.summary()

# 对测试数据进行评估并打印结果
# 评估模型在测试数据上的性能，并返回评估结果，包括损失值和准确率。具体来说，
# 它计算模型在给定测试数据 x_test 和对应标签 y_test 上的损失值和准确率，并将这些值作为一个列表返回。
print(model.evaluate(x_test, y_test))

# #------------------------------【预测】---------------------------------

# 设置要查看的图像索引
image_index = 123
# 显示图像
plt.imshow(x_test[image_index])
plt.show()
# 重新调整输入图像的形状
mod_input = x_test[image_index].reshape(1, 24, 32, 1)
print(mod_input.shape)
# 使用模型进行预测
pred = model.predict(mod_input)
print("实际数字为: ", y_test[image_index])
print("预测数字为: ", pred.argmax())

