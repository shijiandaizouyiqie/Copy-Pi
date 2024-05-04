# serial用于串口通信，numpy用于数据处理，tensorflow用于加载模型和进行预测。
import serial
import numpy as np
import tensorflow as tf

# 配置串口：打开COM10端口，设置波特率为115200，并将超时设置为None，即不设置超时。
ser = serial.Serial('COM10', baudrate=115200, timeout=None)

# 初始化变量：准备两个空列表，用于存储数据和对应的标签。
data_list = []
y_train_list = []

# 加载模型：使用tf.keras.models.load_model方法加载预训练好的手势识别模型gesture.h5
model = tf.keras.models.load_model('2.software\\2.Advanced\\3.Thermal-camera-gesture-recognition\\python_codes\\gesture.h5')
try:

    while 1:
        #从串口ser读取一行数据，然后将其解码为UTF-8格式的字符串(会被begin和end标识符包裹)，并去除字符串两端的空白字符（如空格、制表符、换行符等）
        line = ser.readline().decode('utf-8').strip()
        # 检查数据是否完整
        if line.startswith('begin,') and line.endswith('end,'):
            # 提取data部分并转换为浮点数列表

            # 截取出'begin,'和'end,\r\n'之间的部分，即数据部分，并去除两端的空白字符
            data_str = line[len('begin,'):-len('end,\r\n')].strip()
            # 将data_str以逗号为分隔符拆分成一个浮点数列表data_float。
            data_float = [float(num) for num in data_str.split(',')]
            # 将data_float转换为NumPy数组x_test。
            x_test = np.array(data_float)
            # 计算x_test中的最大值和最小值，分别存储在max_value和min_value中
            max_value = np.max(x_test)
            min_value = np.min(x_test)

            # 最大最小值归一化操作
            x_train_normalized = (x_test - min_value) / (max_value - min_value)
            x_test = x_train_normalized.reshape(1, 24, 32, 1)  #数据维度转换

            pred = model.predict(x_test)
            # model.predict会将输入数据（这里是x_test）传入模型中，模型根据已学到的权重和偏置进行计算，最终得到输出。
            # 对于分类问题，输出通常是一个概率分布，表示每个类别的概率；对于回归问题，输出通常是一个连续值。
            print("预测数字为: ",pred.argmax())
            

except KeyboardInterrupt:
    # 中断时执行的代码
    pass

finally:
    # 关闭串口
    ser.close()


# 通过串口接收数据：进入一个无限循环，持续从串口读取数据。
# 数据处理和预测：当接收到完整的数据时，提取数据部分并转换为浮点数列表。对数据进行最大最小值归一化处理，并将其转换为模型所需的输入格式（1个样本，24x32大小的图像）。
# 使用模型进行预测：调用模型的predict方法对处理后的数据进行预测，得到预测结果。
# 打印预测结果：输出预测的手势类别。
# 异常处理和串口关闭：捕获键盘中断异常（Ctrl+C），关闭串口。


# 对于图像分类问题，输入一张图像后，模型会经过一系列的卷积、池化、激活等操作，
# 最终得到一个包含各个类别的概率分布。然后，模型会选择具有最高概率的类别作为预测结果输出。

# 在目标检测中，最终得到的输出不仅包括每个类别的概率，还包括检测到的物体的位置信息，
#通常表示为矩形边界框。每个类别都有一个对应的名称，通过将类别标签与名称对应起来，可以输出识别到的物体的名称。
# 模型会输出每个检测到的物体的类别和位置信息。然后，针对每个类别，在具有最高概率的位置信息上标注该类别，从而确定检测到的物体的类别。
# 通常，还会输出每个检测到的物体的置信度（即概率），以便于确定检测结果的可靠性。