from Tools_self.tools_self import ToolsSelf
from ConvDefine_self.unet_self import unet_self
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = unet_self((64, 64, 3))

# 指定数据存放的文件夹，其中包括的应该是文件夹，包括的文件夹里面放置的是图片数据
path = r'../DataSet_self'
# 这句代码将这个文件夹下的所有文件夹名字保存下来
dir_list = ToolsSelf.generate_dir_list(path=path)

# 循环某文件夹下的所有数据
# range()函数中的数字表示需要对数据进行多少次训练
# 列表里面存放训练的损失和评估等信息
history_list = []
# for i in range(5):
for each in dir_list:
    data = ToolsSelf.get_data(path=each)
    label = ToolsSelf.get_label_data(path=each)

    # 如果需要更改输入图片的大小，更改这两句的值即可，但是长宽必须是相等的且是32的倍数
    # 如果更改的话，两个都需要改，不然会发生输入和输出尺寸不匹配
    data = ToolsSelf.resize_image(data, (64, 64))
    label = ToolsSelf.resize_image(label, (64, 64))

    print(data.shape)
    print(label.shape)
    history = model.fit(data, label, batch_size=20, epochs=10)
    history_list.append(history)

# 这句保存训练好的模型
model.save("model")
total_loss = np.array(history_list[0].history['loss']) +\
             np.array(history_list[1].history['loss']) +\
             np.array(history_list[2].history['loss']) +\
             np.array(history_list[3].history['loss'])

total_loss = total_loss / 4
plt.plot(total_loss)
plt.show()

total_acc = np.array(history_list[0].history['acc']) + \
            np.array(history_list[1].history['acc']) + \
            np.array(history_list[2].history['acc']) + \
            np.array(history_list[3].history['acc'])

total_acc = total_acc / 4
plt.plot(total_acc)
plt.show()
# 指定需要进行预测的图片
predict_path = [
    r'../test_data/Ex.Aa_f=0.00625[Hz]_Z=15[m].tiff',
    r'../test_data/Ex.Aa_f=0.00625[Hz]_Z=20[m].tiff',
    r'../test_data/Hz.Ta_f=58775.281[Hz]_Z=25[m].tiff',
]

# 以下代码是将需要进行预测的图片进行处理，处理后进行预测和显示
for path in predict_path:
    data_array = cv2.imread(path).astype('float32')
    data_array = cv2.cvtColor(data_array, cv2.COLOR_BGR2RGB) / 255
    new_size_image = cv2.resize(data_array, (64, 64))
    pre = model.predict(np.array([new_size_image]))
    # plt.axis('off')
    plt.imshow(pre[0])
    plt.show()