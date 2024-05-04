import os
import numpy as np
import matplotlib.pyplot as plt

# 创建保存图像的文件夹
output_folder = 'images'
os.makedirs(output_folder, exist_ok=True)

# 加载.npz文件
data = np.load('D:/Visual Studio Code/Open-source-projects/FryPi/2.software/2.Advanced/3.Thermal-camera-gesture-recognition/python_codes/camera_data/train_data.npz')

# 获取数据
x_train = data['x_train']
y_train = data['y_train']
x_train = x_train.reshape(x_train.shape[0], 24, 32)

# 遍历每个样本
for i in range(x_train.shape[0]):
    # 创建图像文件名，可以根据需要调整文件名格式
    image_filename = os.path.join(output_folder, f"sample_{i+1}_label_{y_train[i]}.png")

    # 显示图像并保存
    plt.imshow(x_train[i], cmap='viridis')  # 假设x_train的每个元素是一个2D数组
    plt.title(f"Sample {i+1}, Label {y_train[i]}")
    plt.savefig(image_filename)
    plt.close()

# 关闭文件
data.close()

# 这段代码会将train_data.npz文件中的图像数据保存为PNG格式的图像文件，并将这些文件存储在名为images的文件夹中。每个图像文件的命名格式为sample_{样本编号}_label_{标签值}.png。
# data_2_imgfile.py用于在本地创建一个文件夹，里面存放train_data.npz训练集的图像数据。

