import numpy as np
import matplotlib.pyplot as plt

# 加载.npz文件
data = np.load('D:/Visual Studio Code/Open-source-projects/FryPi/2.software/2.Advanced/3.Thermal-camera-gesture-recognition/python_codes/camera_data/test_data.npz')

# 获取数据
# 训练集
# x_test = data['x_train']
# y_test = data['y_train']

# 测试集
x_test = data['x_test']
y_test = data['y_test']

# 查看数据形状
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# 将一维数组转换为二维数组（32x24）
two_dimensional_array = x_test.reshape(x_test.shape[0], 24, 32)

# 绘制图表
image_index = 0 #起始图像索引
while image_index < x_test.shape[0]:
    plt.imshow(two_dimensional_array[image_index], cmap='viridis')
    plt.title("2D Array Visualization")
    plt.show()
    print("label:",y_test[image_index],"type:",type(y_test[image_index]))
    image_index+=1



# 这段代码用于查看数据集文件中的图像数据以及其对应的标签。它加载了文件，将图像数据转换为二维数组，并逐个展示每张图像及其标签。
# 通过这种方式，可以可视化训练数据集中的图像数据。

# 这个代码用于在终端中查看训练数据集中的图像数据，并可视化其标签。
#