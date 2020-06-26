import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ToolsSelf:
    """
    :parameter  path: 传递存放数据的文件夹路径
                use_gray: 控制是否使用灰度图像，True为使用，False为不使用
    """
    @staticmethod
    def get_data(path: str, use_gray=False) -> np.ndarray:
        print("Getting data from: ", path)
        # 文件名列表
        data_name_list = os.listdir(path=path)
        # print(data_name_list)
        # 数据列表
        data_list = []
        for each in data_name_list:
            if each.find("label") == -1:
                data_array = cv2.imread(path + '/' + each, 1).astype('float32')
                data_array = cv2.cvtColor(data_array, cv2.COLOR_BGR2RGB)
                if use_gray:
                    data_array = cv2.cvtColor(data_array, cv2.COLOR_RGB2GRAY)
                    data_array = cv2.cvtColor(data_array, cv2.COLOR_GRAY2RGB)
                data_list.append(data_array)
        return np.array(data_list) / 255

    """
    :parameter  path: 传递存放数据的文件夹路径
                use_gray: 控制是否使用灰度图像，True为使用，False为不使用
    """
    @staticmethod
    def get_label_data(path: str, use_gray=False) -> np.ndarray:
        print("Getting label from: ", path)
        # 文件名列表
        data_name_list = os.listdir(path=path)
        # 数据列表
        data_list = []
        for each in data_name_list:
            if each.find("label") != -1:
                data_array = cv2.imread(path + '/' + each, 1).astype('float32')
                data_array = cv2.cvtColor(data_array, cv2.COLOR_BGR2RGB)
                if use_gray:
                    data_array = cv2.cvtColor(data_array, cv2.COLOR_RGB2GRAY)
                    data_array = cv2.cvtColor(data_array, cv2.COLOR_GRAY2RGB)
                data_list.append(data_array)
        return np.array(data_list) / 255

    @staticmethod
    def generate_dir_list(path: str) -> list:
        print("Getting data from: ", path)
        path_list = os.listdir(path=path)
        dir_list_self = []
        for each in path_list:
            if os.path.isdir(path + '/' + each):
                dir_list_self.append(path + '/' + each)
        return dir_list_self

    @staticmethod
    def get_shape(input_data: np.ndarray) -> tuple:
        global dim
        shape = [input_data.shape[1], input_data.shape[2]]
        try:
            dim = input_data.shape[3]
        except IndexError:
            shape.append(1)
        shape.append(dim)
        return tuple(shape)

    @staticmethod
    def resize_image(images: np.ndarray, shape: tuple, use_gray=False) -> np.ndarray:
        new_images = []
        for each in images:
            new_size_image = cv2.resize(each, shape)
            if use_gray:
                new_size_image = cv2.cvtColor(new_size_image, cv2.COLOR_RGB2GRAY)
                new_size_image = new_size_image.reshape((shape[0], shape[1], 1))
            new_images.append(new_size_image)
        return np.array(new_images)


# 测试函数的可用性
if __name__ == "__main__":
    data = ToolsSelf.get_data(r'E:\Archive\EM3D-22-Mar-2020.17-47-11\ARfields\22-Mar-2020.17-51-15\Depth(y)=40')
    data_label = ToolsSelf.get_label_data(
        r'E:\Archive\EM3D-22-Mar-2020.17-47-11\ARfields\22-Mar-2020.17-51-15\Depth(y)=40')
    print(data.shape)
    print(data_label.shape)
    print(ToolsSelf.get_shape(data_label))
    # plt.imshow(data[0])
    # plt.show()
    # plt.imshow(data_label[0])
    # plt.show()
    # dir_list = ToolsSelf.generate_dir_list(r'E:\Archive\EM3D-22-Mar-2020.17-47-11\ARfields\22-Mar-2020.17-51-15')
    # for each in dir_list:
    #     print(each)