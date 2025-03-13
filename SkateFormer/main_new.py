# tensorboard --logdir=work_dir/ma52/SkateFormer_j/runs

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix, f1_score
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction

# LR Scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

contrast = True
head_only = False
print(f"contrast:{contrast}, head_only:{head_only}")


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class AdaptiveLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        """
        :param smoothing: 初始平滑系数 (Initial smoothing factor)
        """
        super(AdaptiveLabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        # x: 模型的原始输出 (logits) | x: model's raw outputs (logits)
        # target: 目标类别标签 | target: ground truth labels

        # 计算预测概率以及目标类别的置信度
        # Compute the probabilities and the confidence of the target class
        pred = F.softmax(x, dim=-1)
        confidence = pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)  # shape: (batch,)

        # 根据目标类别的置信度调整平滑系数
        # Adaptive smoothing: higher confidence -> lower smoothing; lower confidence -> higher smoothing
        adaptive_smoothing = self.smoothing * (1 - confidence)  # shape: (batch,)

        # 计算log概率
        # Compute log probabilities
        logprobs = F.log_softmax(x, dim=-1)
        # 计算负对数似然损失 (Negative Log-Likelihood loss)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        # 计算平滑损失 (Smooth loss)，即所有类别的均值
        smooth_loss = -logprobs.mean(dim=-1)

        # 结合两部分损失，注意adaptive_smoothing是逐样本的
        # Combine losses: weighted sum of NLL loss and smooth loss
        loss = (1 - adaptive_smoothing) * nll_loss + adaptive_smoothing * smooth_loss

        return loss.mean()

def augment(x, amp=0.01):
    """
    数据增强函数 (Data Augmentation)
    对输入骨架数据加入随机噪声
    输入 x 形状: [B, C, T, V, M]
    """
    noise = torch.randn_like(x) * amp  # 噪声幅度 (noise amplitude) 可调整
    return x + noise

def augment2(x, amp1=0.01, amp2=0.01):
    """
    数据增强函数 (Data Augmentation)
    对输入骨架数据进行线性变换扭曲 (Apply linear transformation distortion)
    仅对 C 的前两个维度进行变换 (Only apply transformation on the first two channels)
    输入 x 形状: [B, C, T, V, M]
    其中 C 为归一化到 [-1,1] 的 xy 坐标 (where C represents normalized xy coordinates)
    """
    B, C, T, V, M = x.shape
    # 提取前两个通道用于变换 (Extract first two channels for transformation)
    x_first = x[:, :2, :, :, :]   # shape: [B, 2, T, V, M]
    # 如果存在其他通道，则保留 (Keep remaining channels unchanged)
    x_rest = x[:, 2:, :, :, :] if C > 2 else None

    # 为每个样本的每一帧生成初始为单位矩阵的变换矩阵 (Generate identity matrices for each frame)
    A = torch.eye(2, device=x.device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)  # shape: [B, T, 2, 2]
    # 为每一帧加入随机扰动 (Add random perturbation to each frame)
    A = A + torch.randn(B, T, 2, 2, device=x.device) * amp1

    # 调整 x_first 的维度，使 xy 坐标维度放到最后 (Permute dimensions so that xy coordinates are last)
    x_first_perm = x_first.permute(0, 2, 3, 4, 1)  # shape: [B, T, V, M, 2]
    # 对每一帧的 xy 坐标进行线性变换 (Apply linear transformation to xy coordinates)
    x_first_transformed = torch.einsum('btij,btvmi->btvmi', A, x_first_perm)
    # 添加随机噪声 (Add random noise)
    x_first_transformed = x_first_transformed + torch.randn_like(x_first_transformed) * amp2
    # 将维度还原回原始顺序 (Permute dimensions back to original order)
    x_first_transformed = x_first_transformed.permute(0, 4, 1, 2, 3)  # shape: [B, 2, T, V, M]

    # 如果存在其他通道，将变换后的部分和未变换部分拼接 (Concatenate transformed and unchanged channels)
    if x_rest is not None:
        x_transformed = torch.cat([x_first_transformed, x_rest], dim=1)  # 在 C 维度上拼接 (concatenate along channel dim)
    else:
        x_transformed = x_first_transformed

    return x_transformed



def augment3(x, amp1=0.01, amp2=0.01):
    """
    数据增强函数 (Data Augmentation)
    对输入骨架数据进行线性变换扭曲 (Apply linear transformation distortion)
    输入 x 形状: [B, C, T, V, M]
    其中 C 为归一化到 [-1,1] 的 xy 坐标 (where C represents normalized xy coordinates)
    每连续6个节点为一个身体部位, 共8个身体部位 (Every consecutive 6 nodes represent one body part, total 8 parts)
    每次仅对一个随机选取的身体部位进行变换 (Apply transformation only to one randomly selected body part)
    """
    B, C, T, V, M = x.shape

    # 随机选择一个身体部位的索引 (randomly choose a body part index from 0 to 7)
    part_idx = torch.randint(0, 8, (1,)).item()
    start_node = part_idx * 6
    end_node = start_node + 6

    # 为每个样本的每一帧生成初始为单位矩阵的变换矩阵 (Create identity matrices for each sample and frame)
    # 形状为 [B, T, 2, 2]
    A = torch.eye(2, device=x.device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
    # 为每一帧加入随机扰动，amp1 控制扰动幅度 (Add random perturbation to each frame, controlled by amp1)
    A = A + torch.randn(B, T, 2, 2, device=x.device) * amp1

    # 调整 x 的维度，将 xy 坐标维度移到最后 (permute dimensions: from [B, C, T, V, M] to [B, T, V, M, C])
    x_perm = x.permute(0, 2, 3, 4, 1)

    # 复制原始数据 (copy the original data)
    x_transformed = x_perm.clone()

    # 对选定的身体部位进行线性变换 (apply linear transformation to the selected body part)
    x_transformed[:, :, start_node:end_node, :, :] = torch.einsum('btij,btvmi->btvmi', A,
                                                                  x_perm[:, :, start_node:end_node, :, :])
    # 添加随机噪声，amp2 控制噪声幅度 (add random noise, controlled by amp2)
    x_transformed[:, :, start_node:end_node, :, :] += torch.randn_like(
        x_transformed[:, :, start_node:end_node, :, :]) * amp2

    # 将维度还原回原始顺序 (permute back to original shape: [B, C, T, V, M])
    x_transformed = x_transformed.permute(0, 4, 1, 2, 3)
    return x_transformed, part_idx


class NTXentLoss(nn.Module):
    """
    NT-Xent 损失 (NT-Xent Loss / InfoNCE Loss) 用于对比学习
    """

    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().to(device)

    def _get_correlated_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, i + self.batch_size] = 0
            mask[i + self.batch_size, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        计算 NT-Xent 损失 (Compute NT-Xent Loss)
        :param z_i: 第一个 view 嵌入 [B, dim]
        :param z_j: 第二个 view 嵌入 [B, dim]
        :return: 损失标量 (loss scalar)
        """
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)  # [2B, dim]
        z = F.normalize(z, dim=1)
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        pos_sim = torch.cat([
            torch.diag(similarity_matrix, self.batch_size),
            torch.diag(similarity_matrix, -self.batch_size)
        ]).view(N, 1)
        neg = similarity_matrix[self.mask].view(N, -1)
        logits = torch.cat((pos_sim, neg), dim=1)
        labels = torch.zeros(N, dtype=torch.long).to(self.device)
        loss = self.criterion(logits, labels)
        return loss / N


def get_parser():
    parser = argparse.ArgumentParser(
        description='SkateFormer: Skeletal-Temporal Trnasformer for Human Action Recognition')
    parser.add_argument('--work-dir', default=None, help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')

    # ===config==========================================================================================================================================
    # parser.add_argument('--config', default='./config/SkateFormer_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_s_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_newPE_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='config/SkateFormer_j_-TGconv_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='config/SkateFormer_j_-TGconv2_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_newFusion_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_4w_SM_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_4w_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_4w_s_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_6w_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_6w_s_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_MKT_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_MoE_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_MoE_simple_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_MoE_Linear_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_SE_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_SE2_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_SE3_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_CC_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_6w_s_newPE_j_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_CrossAttention_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_CrossAttention2_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_CrossAttention3_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_my_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_2MHA_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_ds_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_ds_6w_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_j_NEW_c2.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_6w_s_j_NEW_c_new.yaml', help='path to the configuration file')
    # parser.add_argument('--config', default='./config/SkateFormer_6w_s_j_NEW_c_new_sparise.yaml', help='path to the configuration file')
    parser.add_argument('--config', default='./config/SkateFormer_6w_s_j_NEW_c_new_Taylor.yaml', help='path to the configuration file')
    # ===config==========================================================================================================================================
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    # parser.add_argument('--weights', default="./runs-49-7644.pt", help='the weights for network initialization')
    # parser.add_argument('--weights', default="./runs-26-18278.pt", help='the weights for network initialization')
    # parser.add_argument('--weights', default="./xxx_61_80.pt", help='the weights for network initialization')  # 60.67%
    # parser.add_argument('--weights', default="./xxx_62_30.pt", help='the weights for network initialization')
    # ===config==========================================================================================================================================

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False,
                        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default=None, help='data loader will be used')
    parser.add_argument('--num-worker', type=int, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    # parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--min-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warmup-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warmup_prefix', type=bool, default=False)
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--grad-clip', type=bool, default=False)
    parser.add_argument('--grad-max', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='AdamW', help='type of optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', help='type of learning rate scheduler')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-ratio', type=float, default=0.001, help='decay rate for learning rate')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--loss-type', type=str, default='CE')
    return parser


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')

        self.global_step = 0
        self.load_model()
        self.load_data()

        if self.arg.phase == 'train':
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader['train']))

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
        else:
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)
            # self.loss = AdaptiveLabelSmoothing(smoothing=0.1).cuda(output_device)
            self.contrast_loss = NTXentLoss(batch_size=arg.batch_size, temperature=0.5, device=output_device)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model.load_state_dict(weights, strict=False)
                if head_only:
                    for param in self.model.parameters():
                        param.requires_grad = False
                    for param in self.model.head.parameters():
                        param.requires_grad = True
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state, strict=False)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def load_scheduler(self, n_iter_per_epoch):
        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)

        self.lr_scheduler = None
        if self.arg.lr_scheduler == 'cosine':
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=(num_steps - warmup_steps) if self.arg.warmup_prefix else num_steps,
                lr_min=self.arg.min_lr,
                warmup_lr_init=self.arg.warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=self.arg.warmup_prefix,
            )
        else:
            raise ValueError()

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def show_data(self, data, index, part_idx):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        inward_ori_index = [
            (43, 42),
            (13, 42),
            (19, 42),
            (13, 15),
            (15, 17),
            (19, 21),
            (21, 23),
            (46, 42),
            (42, 47),
            (47, 24),
            (47, 30),
            (24, 25),
            (30, 31),
            (25, 26),
            (31, 32),
            (29, 27),
            (35, 33),
            (34, 33),
            (28, 27),
            (26, 27),
            (32, 33),
            (17, 0),
            (0, 1),
            (0, 2),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (23, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (38, 43),
            (37, 36),
            (38, 37),
            (39, 38),
            (39, 40),
            (39, 41)
        ]

        # 假设 data 的形状为 (BS, C, T, V, 1)
        # Assume data has shape (BS, C, T, V, 1)
        # 只选取第一个 batch (BS=0)
        data_sample = data[0, :, :, :, 0].cpu().numpy()  # (C, T, V)
        # 其中 C=2 为坐标 (coordinates)，T=64 为帧数 (frames)，V 为关节点数 (joints)
        C, T, V = data_sample.shape

        # 创建绘图对象 (create figure and axis)
        fig, ax = plt.subplots(figsize=(6, 6))

        # 初始化函数，用于动画的第一帧 (Initialization function for animation)
        def init():
            ax.clear()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_xticks([])
            ax.set_yticks([])
            return []

        # 更新函数，每一帧都会调用此函数 (Update function called for each frame)
        def update(t):
            ax.clear()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Frame {t}')
            # 绘制连线 (Plot skeleton links)
            for (j1, j2) in inward_ori_index:
                # 取反坐标以便显示 (invert coordinates for display)
                x1, y1 = -data_sample[0, t, j1], -data_sample[1, t, j1]
                x2, y2 = -data_sample[0, t, j2], -data_sample[1, t, j2]
                ax.plot([x1, x2], [y1, y2], c='b', linewidth=1)  # 蓝色线段 (blue line)
            # 绘制关节点 (Plot joints)
            ax.scatter(-data_sample[0, t, :], -data_sample[1, t, :], c='r', marker='o')
            return []

        # 创建动画 (Create animation)
        ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False)

        # 保存动画为 GIF，使用 PillowWriter
        # Save animation as a gif using PillowWriter
        ani.save(f'index{int(index[0])}_part_idx{int(part_idx)}_{time.time()}.gif', writer='pillow', fps=32)

        plt.show()

    def map_tensor_labels(self, labels):
        mapping = {range(0, 5): 0, range(5, 11): 1, range(11, 24): 2, range(24, 32): 3, range(32, 38): 4,
                   range(38, 48): 5, range(48, 52): 6}
        return torch.tensor(
            [next((v for r, v in mapping.items() if int(label) in r), None) for label in labels],
            device=labels.device)

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        for batch_idx, (data, index_t, label, index) in enumerate(process):
            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)  # 64,2,64,28,1 BS,C,T,V,1
                index_t = index_t.float().cuda(self.output_device)  # 64,64 BS,T
                label = label.long().cuda(self.output_device)  # 64, BS
            timer['dataloader'] += self.split_time()

            # forward output_forward_head, output_projection_head
            x1 = augment2(data)  # Shape: [B, C, T, V, M]
            x2 = augment2(data)  # Shape: [B, C, T, V, M]
            # self.show_data(data, index, 0)
            # self.show_data(x1, index, part_idx1)
            # self.show_data(x2, index, part_idx2)

            big_batch = torch.cat([data, x1, x2], dim=0)  # Shape: [3*B, C, T, V, M]
            big_batch_index_t = torch.cat([index_t, index_t, index_t], dim=0)  # Shape: [3*B, T]

            output_forward_head, output_projection_head, output_coarse_heads = self.model(big_batch, big_batch_index_t)
            B = data.size(0)  # Get the batch size
            # Split classification outputs
            output = output_forward_head[:B]  # Original data output, shape [B, num_classes]
            output_coarse = [o[:B] for o in output_coarse_heads]  # Original data output, shape [B, num_classes]
            # Split contrastive features
            z1 = output_projection_head[B:2 * B]  # x1 features, shape [B, projection_dim]
            z2 = output_projection_head[2 * B:]  # x2 features, shape [B, projection_dim]

            coarse_labels = self.map_tensor_labels(label)
            # loss = (self.loss(output, label) + self.contrast_loss(z1, z2) +
            #         0.25 * self.loss(output_coarse[-1], coarse_labels) +
            #         0.1 * self.loss(output_coarse[-2], coarse_labels) +
            #         0.05 * self.loss(output_coarse[-3], coarse_labels) +
            #         0.01 * self.loss(output_coarse[-4], coarse_labels))
            loss = 0.5 * self.loss(output, label) + 0.5 * self.contrast_loss(z1, z2)
            # loss = self.loss(output, label)
            # loss = self.loss(output_coarse, coarse_labels)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.arg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_max)
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)

            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()



        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tMean training acc: {:.2f}%.'.format(np.mean(acc_value) * 100))
        self.print_log('\tLearning Rate: {:.8f}'.format(self.lr))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights,
                       self.arg.model_saved_name + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')
            self.print_log(f"\tSaved model checkpoint to {self.arg.model_saved_name + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt'}")

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, index_t, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    index_t = index_t.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output, _, output_coarse = self.model(data, index_t)

                    loss = self.loss(output, label)
                    loss_value.append(loss.data.item())

                    score_frag.append(output.data.cpu().numpy())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ma52' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss_fine of {} batches: {}.'.format(ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}_fine: {:.2f}%'.format(k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            self.eval(-1, save_score=True, loader_name=['test'])
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch, save_model=True)
                self.eval(epoch, save_score=True, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

