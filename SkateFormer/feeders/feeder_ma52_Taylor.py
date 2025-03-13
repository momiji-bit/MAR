import numpy as np
import json
import random
import math
import pickle
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support


class Feeder(Dataset):
    def __init__(self, data_path, label_path, data_type='j', repeat=1, p=0.5,
                 window_size=-1, debug=False, partition=False):
        """
        初始化Feeder类，用于加载和预处理骨架数据。

        参数:
            data_path (str): 数据文件路径。
            label_path (str): 标签文件路径。
            data_type (str): 数据类型，默认为'j'。
            repeat (int): 数据重复次数，默认为1。
            p (float): 随机丢弃的概率，默认为0.5。
            window_size (int): 窗口大小，默认为-1。
            debug (bool): 是否开启调试模式，默认为False。
            partition (bool): 是否进行身体部位划分，默认为False。
        """

        # 判断是否为验证集，根据label_path中的'val'关键字
        if 'val' in label_path:
            self.train_val = 'val'
            
            with open('data/MA52-pose151/val_label.pkl', 'rb') as f:
            # with open('data/MA52-pose151/test_label.pkl', 'rb') as f:
                self.data_dict = pickle.load(f)
        else:
            self.train_val = 'train'
            with open('data/MA52-pose151/train_label.pkl', 'rb') as f:
                self.data_dict = pickle.load(f)

        # 设置数据根目录
        self.time_steps = 64  # 时间步长

        self.label = []
        self.label_coarse = []
        # 提取每个数据样本的标签，并将标签存储在self.label列表中
        for index in range(len(self.data_dict)):
            info = self.data_dict[index]
            self.label.append(int(info['label']))

        mapping = {range(0, 5): 0, range(5, 11): 1, range(11, 24): 2, range(24, 32): 3, range(32, 38): 4,
                   range(38, 48): 5, range(48, 52): 6}
        self.label_coarse = [next((v for r, v in mapping.items() if int(label) in r), None) for label in self.label]

        # 保存初始化参数
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.data_type = data_type
        self.window_size = window_size
        self.partition = partition
        self.repeat = repeat
        self.p = p

        RH = 94
        LH = 115
        FK = 26
        # 定义骨骼连接关系，列表中的元组表示骨骼的连接点
        self.bone = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Body
            (0, 18),
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            # Foot
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),
            # Right Hand
            (9, RH+0), (RH+0, RH+4), (RH+0, RH+8), (RH+0, RH+8), (RH+0, RH+12), (RH+0, RH+16), (RH+0, RH+20),
            # Left Hand
            (10, LH + 0), (LH + 0, LH + 4), (LH + 0, LH + 8), (LH + 0, LH + 12), (LH + 0, LH + 16), (LH + 0, LH + 20),
            # Face
            (FK + 33, 0), (FK + 51, FK + 57), (FK + 33, FK + 51), (FK + 28, FK + 33), (FK + 28, FK + 45), (FK + 28, FK + 36)
        ]

        # 如果需要进行身体部位划分，定义各部分的关节索引
        if self.partition:
            self.right_hand = np.array([RH+0, RH+4, RH+8, RH+12, RH+16, RH+20])
            self.left_hand = np.array([LH+0, LH+4, LH+8, LH+12, LH+16, LH+20])

            self.right_arm = np.array([5,5,7,7,9,9])
            self.left_arm = np.array([6,6,8,8,10,10])

            self.right_leg = np.array([11,13,15,24,22,20])
            self.left_leg = np.array([12,14,16,25,23,21])

            self.face = np.array([FK+57, FK+51, FK+33, FK+28, FK+45, FK+36])
            self.torso = np.array([18,0,3,4,17,19])

            self.new_idx = np.concatenate(
                (self.right_hand, self.left_hand, self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.face, self.torso), axis=-1)

            with open('data/MA52-pose151/train_new.pkl', 'rb') as f:
                self.train_new = pickle.load(f)
            with open('data/MA52-pose151/val_new.pkl', 'rb') as f:
            # with open('data/MA52-pose151/test_new.pkl', 'rb') as f:
                self.val_new = pickle.load(f)

            self.xxx = {}
            # 合并两个字典时可以使用字典解包操作
            a = {i['frame_dir']: i['keypoint'][0] for i in self.train_new['annotations']}
            b = {i['frame_dir']: i['keypoint'][0] for i in self.val_new['annotations']}
            # 合并字典
            self.xxx = {**a, **b}

            # 加载数据
            self.load_data()

    def taylor_video(self, videoinput, terms=3, temporal_block=4):
        """
        对输入的视频骨架序列进行Taylor级数变换 (Apply Taylor series transformation to the skeleton video sequence).

        参数 (Parameters):
          videoinput: 存储骨架序列的mat文件路径 (the path of the mat file containing skeleton sequence)
          terms: Taylor级数展开项数 (number of Taylor series terms)
          temporal_block: 时间块的长度 (length of the temporal block)
        """

        def factorial(n):
            """
            使用递归计算阶乘 (calculate factorial recursively).
            基本情况：0的阶乘为1 (base case: factorial of 0 is 1)
            """
            if n == 0:
                return 1
            else:
                return n * factorial(n - 1)
        video = np.transpose(videoinput, (1, 2, 0))
        # 获取骨架、通道、时间帧的数量 (Joints, Channels, Time)
        J, C, T = video.shape

        if temporal_block - 1 < terms:
            print('给定的时间块长度不足以计算定义的级数项 (The given temporal block length is not enough to compute defined terms).')
            return None
        else:
            # 初始化 Taylor 变换后的视频数组
            Taylor = np.zeros((J, C, T - temporal_block + 1))

            # 对每个时间块进行操作 (process each temporal block)
            for i in range(T - temporal_block + 1):
                # 获取当前时间块内的骨架数据 (extract video clip for current block)
                video_clip = video[:, :, i:i + temporal_block]  # shape: (J, C, temporal_block)
                # 取当前时间块第一帧数据，并在第三个维度扩展 (get the first frame and repeat to match temporal block size)
                slice_tensor = video[:, :, i][:, :, np.newaxis]  # shape: (J, C, 1)
                dummy_clip = np.repeat(slice_tensor, temporal_block, axis=2)  # shape: (J, C, temporal_block)

                # 计算时间差分 (compute temporal differences)
                delta_temp = video_clip - dummy_clip  # shape: (J, C, temporal_block)

                # 初始化存储各阶差分的数组 (store differences for each order)
                D_temp = np.zeros((J, C, terms))
                temp = video_clip.copy()

                # 计算 Taylor 展开中各阶的差分 (compute differences for each Taylor term)
                for j in range(terms):
                    diff = temp[:, :, 1:] - temp[:, :, :-1]  # 计算差分 (compute difference)
                    # 这里仅取第一帧的差分作为代表 (use the first difference of the current order)
                    D_temp[:, :, j] = diff[:, :, 0]
                    temp = diff

                # 初始化矩阵 M，用于存放 Taylor 级数展开结果 (accumulate Taylor series expansion)
                M = np.zeros((J, C, temporal_block))
                # 累加 Taylor 展开各阶项 (accumulate each order term)
                for order in range(terms):
                    # 使用阶乘和差分，结合 delta_temp 的幂次 (combine factorial, difference and powers of delta)
                    # 注意: 使用 reshape 保证广播操作 (ensure correct broadcasting)
                    term_value = (D_temp[:, :, order][:, :, np.newaxis] / factorial(order)) * np.power(delta_temp, order)
                    M = M + term_value

                # 计算当前时间块的平均值作为 Taylor 变换后的帧 (compute average over temporal block)
                taylor_frame = np.mean(M, axis=2)  # shape: (J, C)
                Taylor[:, :, i] = taylor_frame

            Taylor = np.transpose(Taylor, (2, 0, 1))
            return Taylor

    def load_data(self):
        """
        加载所有数据样本的骨架信息。
        数据格式为N C V T M，其中N是样本数，C是坐标维度，V是关节数，T是时间步长，M是人体数。
        """
        # data: N C V T M
        self.data = []

        for data in tqdm(self.data_dict):
            file_name = data['file_name']
            xxx = self.xxx[file_name]
            taylor_xxx = self.taylor_video(xxx)
            stack_xxx = np.concatenate([xxx[:-3,:,:], taylor_xxx], axis=2)
            self.data.append(stack_xxx)

    def __len__(self):
        """
        返回数据集的长度，考虑数据重复次数。
        """
        return len(self.data_dict) * self.repeat

    def __iter__(self):
        """
        返回迭代器自身。
        """
        return self

    def scale_transform_2d(self, X, s):
        # 构造二维缩放矩阵
        Ss = np.asarray([[s, 0],
                         [0, s]])
        # 应用缩放变换
        X0 = np.dot(np.reshape(X, (-1, 2)), Ss)
        # 还原数据形状
        X = np.reshape(X0, X.shape)
        return X


    def show_data(self, value):
        import numpy as np
        import matplotlib.pyplot as plt

        # 选择某个样本进行可视化 (假设 index=0)
        sample = value  # 取第一个样本，形状为 (T, 136, 2)
        T = sample.shape[0]  # 帧数
        V = sample.shape[1]  # 关节点数

        fig, axes = plt.subplots(8, 8, figsize=(50, 50))
        axes = axes.flatten()

        for t in range(min(T, 64)):  # 只显示前 64 帧
            ax = axes[t]
            ax.scatter(-sample[t, :, 0], -sample[t, :, 1], c='r', marker='o')  # 关节点
            ax.set_xlim(-1, 1)  # 调整范围
            ax.set_ylim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Frame {t}')

        plt.tight_layout()
        plt.show()

    def __getitem__(self, index):
        """
        获取指定索引的数据样本。

        参数:
            index (int): 数据索引。

        返回:
            tuple: (数据, 时间索引, 标签, 原始索引)
        """
        channel = 4
        label = self.label[index % len(self.data_dict)]  # 获取标签

        value = self.data[index % len(self.data_dict)]  # 获取数据  T,136,2

        if self.train_val == 'train':
            # 数据增强部分，仅在训练时应用
            random.random()  # 生成一个随机数，用于后续判断

            center = value[0, 19, :]  # 获取中心点（通常是身体的中心）
            value = value - center  # 平移数据，使中心点位于原点

            # 数据归一化到[-1, 1]范围
            scalerValue = np.reshape(value, (-1, channel))
            epsilon = 1e-6  # Small constant to avoid division by zero
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0) + epsilon)
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, value.shape[1], channel))  # 恢复关节数和坐标维度

            # self.show_data(scalerValue)

            data = np.zeros((self.time_steps, value.shape[1], channel))  # 初始化数据数组

            value = scalerValue[:, :, :]
            length = value.shape[0]
            # 随机选择时间步，确保数据长度为time_steps
            random_idx = random.sample(list(np.arange(length)) * self.time_steps, self.time_steps)
            random_idx.sort()
            data[:, :, :] = value[random_idx, :, :]  # 填充数据，形状为(T, V, C)
            index_t = 2 * np.array(random_idx).astype(np.float32) / length - 1  # 时间索引归一化到[-1, 1]

            # # 随机丢弃一个坐标轴的数据
            # if random.random() < self.p:
            #     axis_next = random.randint(0, 1)  # 随机选择一个轴
            #     temp = data.copy()
            #     T, V, C = data.shape
            #     x_new = np.zeros((T, V))
            #     temp[:, :, axis_next] = x_new  # 将选定轴的数据设为0
            #     data = temp

            # 随机丢弃部分关节的数据
            if random.random() < self.p:
                temp = data.copy()
                T, V, C = data.shape
                random_int_v = random.randint(12, 24)  # 随机选择要丢弃的关节数
                all_joints = [i for i in range(V)]
                joint_list_ = random.sample(all_joints, random_int_v)  # 随机选择关节
                joint_list_ = sorted(joint_list_)
                random_int_t = random.randint(16, 32)  # 随机选择要丢弃的时间步数
                all_frames = [i for i in range(T)]
                time_range_ = random.sample(all_frames, random_int_t)  # 随机选择时间步
                time_range_ = sorted(time_range_)
                x_new = np.zeros((len(time_range_), len(joint_list_), C))
                temp2 = temp[time_range_, :, :].copy()
                temp2[:, joint_list_, :] = x_new  # 将选定关节和时间步的数据设为0
                temp[time_range_, :, :] = temp2
                data = temp

        else:
            # 测试或验证集，不进行数据增强
            random.random()

            center = value[0, 19, :]
            value = value - center

            scalerValue = np.reshape(value, (-1, channel))
            epsilon = 1e-6  # Small constant to avoid division by zero
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                    np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0) + epsilon)
            scalerValue = scalerValue * 2 - 1

            # scalerValue = np.reshape(scalerValue, (-1, value.shape[1], 3))
            scalerValue = np.reshape(scalerValue, (-1, value.shape[1], channel))

            # data = np.zeros((self.time_steps, value.shape[1], 3))
            data = np.zeros((self.time_steps, value.shape[1], channel))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            # 使用线性采样来获取固定长度的时间步
            idx = np.linspace(0, length - 1, self.time_steps).astype(np.int)
            data[:, :, :] = value[idx, :, :]
            index_t = 2 * idx.astype(np.float32) / length - 1

        # 根据data_type进行数据类型转换
        if 'b' in self.data_type:
            # 计算骨骼间的相对位置
            data_bone = np.zeros_like(data)
            for bone_idx in range(len(self.bone)):
                # 计算每个骨骼的向量（起点关节 - 终点关节）
                data_bone[:, self.bone[bone_idx][0], :] = (data[:, self.bone[bone_idx][0], :] - data[:, self.bone[bone_idx][1], :])
            data = data_bone

        if 'm' in self.data_type:
            # 计算动作的变化（当前帧与下一帧的差）
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion

        # 转置数据维度为(C, T, V)
        data = np.transpose(data, (2, 0, 1))
        C, T, V = data.shape
        # 增加一个维度，形状变为(C, T, V, 1)
        data = np.reshape(data, (C, T, V, 1))

        # 如果需要进行身体部位划分，选择相应的关节索引
        if self.partition:
            data = data[:, :, self.new_idx]

        return data, index_t, label, index

    def top_k(self, score, top_k, f=False):
        """
        计算top-k准确率。

        参数:
            score (ndarray): 模型预测的分数。
            top_k (int): 选择的k值。

        返回:
            float: top-k准确率。
        """
        rank = score.argsort()  # 对分数进行排序
        # 检查每个样本的真实标签是否在预测的top_k中
        if f:
            hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label_coarse)]
        else:
            hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)



    def f1_macro_micro(self, score, labels):
        """
        计算 F1 Macro 和 F1 Micro 指标。

        参数:
            score (ndarray): 模型预测的分数，shape (n_samples, n_classes)。
            labels (ndarray): 真实标签，长度为 n_samples。

        返回:
            dict: 包含 F1 Macro 和 F1 Micro 的字典。
        """
        # 模型预测的类别为得分最高的索引
        predicted_labels = score.argmax(axis=1)

        # 使用 sklearn 提供的 precision_recall_fscore_support 计算 F1 分数
        precision, recall, f1_scores, _ = precision_recall_fscore_support(
            labels, predicted_labels, average=None
        )

        # F1 Macro: 所有类别的 F1 分数的简单平均
        f1_macro = f1_scores.mean()

        # F1 Micro: 基于全局统计量计算
        _, _, f1_micro, _ = precision_recall_fscore_support(
            labels, predicted_labels, average="micro"
        )

        return f1_macro, f1_micro


def import_class(name):
    """
    动态导入指定的类。

    参数:
        name (str): 类的全路径名称，例如'module.submodule.ClassName'。

    返回:
        class: 导入的类对象。
    """
    components = name.split('.')
    mod = __import__(components[0])  # 导入第一个模块
    for comp in components[1:]:
        mod = getattr(mod, comp)  # 依次获取子模块或类
    return mod
