import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath

""" Partition and Reverse 函数
    这些函数用于对输入张量进行分块（Partition）和逆分块（Reverse），保持多尺度特征处理的一致性
"""


def type_1_partition(input, partition_size):  # partition_size = [N, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_1_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1],
                             -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_2_partition(input, partition_size):  # partition_size = [N, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_2_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1],
                             -1)
    output = output.permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)
    return output


def type_3_partition(input, partition_size):  # partition_size = [M, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_3_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1],
                             -1)
    output = output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_4_partition(input, partition_size):  # partition_size = [M, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_4_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1],
                             -1)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)
    return output


""" 1D Relative Positional Bias """


def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)


""" Multi-Head Self-Attention (MSA) 模块 """


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, rel_type, num_heads=32, partition_size=(1, 1), attn_drop=0., rel=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.rel_type = rel_type
        self.num_heads = num_heads
        self.partition_size = partition_size
        self.scale = num_heads ** -0.5
        self.attn_area = partition_size[0] * partition_size[1]
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rel = rel

        if self.rel:
            if self.rel_type == 'type_1' or self.rel_type == 'type_3':
                self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * partition_size[0] - 1), num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)
                self.ones = torch.ones(partition_size[1], partition_size[1], num_heads)
            elif self.rel_type == 'type_2' or self.rel_type == 'type_4':
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * partition_size[0] - 1), partition_size[1], partition_size[1], num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(self):
        if self.rel_type == 'type_1' or self.rel_type == 'type_3':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], -1)
            relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(3).repeat(1, self.partition_size[1],
                                                                                             1, self.partition_size[1],
                                                                                             1, 1).view(self.attn_area,
                                                                                                        self.attn_area,
                                                                                                        -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)
        elif self.rel_type == 'type_2' or self.rel_type == 'type_4':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], self.partition_size[1], self.partition_size[1], -1)
            relative_position_bias = relative_position_bias.permute(0, 2, 1, 3, 4).contiguous().view(self.attn_area,
                                                                                                     self.attn_area, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)

    def forward(self, input):
        B_, N, C = input.shape
        qkv = input.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rel:
            attn = attn + self._get_relative_positional_bias()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        return output


""" 模块化融合模块 (Modular Fusion Modules) """


class MultiScaleFusionModule(nn.Module):
    """
    多尺度融合模块 (Multi-Scale Fusion Module)
    结合多个分支的特征，通过1x1卷积投影并添加残差连接
    """

    def __init__(self, in_channels, num_branches):
        super(MultiScaleFusionModule, self).__init__()
        self.num_branches = num_branches
        self.alpha = nn.Parameter(torch.ones(num_branches))  # 学习融合权重 (learnable fusion weights)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, branches, skip):
        weights = F.softmax(self.alpha, dim=0)
        fused = 0
        for i in range(self.num_branches):
            fused = fused + weights[i] * branches[i]
        fused = self.proj(fused)
        return fused + skip  # 加入跳跃/残差连接 (add residual connection)


class AttentionFusionModule(nn.Module):
    """
    多注意力分支融合模块 (Attention Fusion Module)
    用于融合来自多个注意力分支的输出
    """

    def __init__(self, in_channels, num_attn_branches):
        super(AttentionFusionModule, self).__init__()
        self.num_attn_branches = num_attn_branches
        self.beta = nn.Parameter(torch.ones(num_attn_branches))

    def forward(self, attn_branches):
        weights = F.softmax(self.beta, dim=0)
        fused = 0
        for i in range(self.num_attn_branches):
            fused = fused + weights[i] * attn_branches[i]
        return fused


""" 改进后的 SkateFormerBlock
    在该模块中，我们对输入先经过归一化和线性映射，然后将通道分为两部分：
    - f_conv 分支用于 G-Conv 和 T-Conv 处理；
    - f_attn 分支用于 Skate-MSA 注意力计算。
    对每个分支的输出，先用 1x1 卷积（Projection）统一到相同通道数，再通过融合模块加权融合，同时加入残差/跳跃连接。
"""


class SkateFormerBlock(nn.Module):
    def __init__(self, in_channels, num_points=50, kernel_size=7, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SkateFormerBlock, self).__init__()
        # 分块函数设置 (Partition functions setup)
        self.type_1_size = type_1_size
        self.type_2_size = type_2_size
        self.type_3_size = type_3_size
        self.type_4_size = type_4_size
        self.partition_function = [type_1_partition, type_2_partition, type_3_partition, type_4_partition]
        self.reverse_function = [type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse]
        self.partition_size = [type_1_size, type_2_size, type_3_size, type_4_size]
        self.rel_type = ['type_1', 'type_2', 'type_3', 'type_4']

        self.norm_1 = norm_layer(in_channels)
        self.mapping = nn.Linear(in_channels, 2 * in_channels, bias=True)

        # G-Conv 分支 (Graph Convolution branch)
        self.gconv = nn.Parameter(torch.zeros(num_heads // 4, num_points, num_points))
        trunc_normal_(self.gconv, std=.02)

        # T-Conv 分支 (Temporal Convolution branch)
        self.tconv = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(kernel_size, 1),
                               padding=((kernel_size - 1) // 2, 0), groups=num_heads // 4)

        # 注意力分支 (Skate-MSA branch)
        attention = []
        for i in range(len(self.partition_function)):
            attention.append(
                MultiHeadSelfAttention(in_channels=in_channels // (len(self.partition_function) * 2),
                                       rel_type=self.rel_type[i],
                                       num_heads=num_heads // (len(self.partition_function) * 2),
                                       partition_size=self.partition_size[i], attn_drop=attn_drop, rel=rel))
        self.attention = nn.ModuleList(attention)

        # 最终融合前的投影层，此处不再依靠通道拼接，而是通过融合模块实现
        self.proj = nn.Linear(in_channels, in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=int(mlp_ratio * in_channels),
                       act_layer=act_layer, drop=drop)

        # 融合模块：内部融合注意力分支 + 融合所有三分支
        self.attn_fusion_module = AttentionFusionModule(in_channels // (len(self.partition_function) * 2),
                                                        len(self.partition_function))
        self.fusion_module = MultiScaleFusionModule(in_channels, 3)

        # 各分支投影层，将各分支输出统一到相同通道数 in_channels
        # 假设各分支输出通道分别为：
        #   G-Conv: (in_channels//4)*(num_heads//4)
        #   T-Conv: in_channels//4
        #   Attention: in_channels//(len(partition)*2)
        # 修改：各分支投影层，统一输出到 in_channels
        # 原来的 proj_gconv 定义错误，应该接受 in_channels//4 个通道（而非 (in_channels//4)*(num_heads//4)）
        self.proj_gconv = nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        self.proj_tconv = nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        # f_attn 分支输出经过分块后，每个注意力分支输出通道数为 3*in_channels/8
        self.proj_attn = nn.Conv2d(in_channels // (len(self.partition_function) * 2), in_channels, kernel_size=1)


    def forward(self, input):
        # 输入形状: [B, C, T, V]
        B, C, T, V = input.shape
        skip = input  # 跳跃/残差连接

        # 预处理：归一化和线性映射 (Normalization and Mapping)
        x = input.permute(0, 2, 3, 1).contiguous()  # [B, T, V, C]
        f = self.mapping(self.norm_1(x))  # [B, T, V, 2C]
        f = f.permute(0, 3, 1, 2).contiguous()  # [B, 2C, T, V]

        # 通道分离：f_conv 用于卷积分支，f_attn 用于注意力分支
        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2], dim=1)

        # G-Conv 分支
        split_f_conv = torch.chunk(f_conv, 2, dim=1)  # 得到两个部分，每部分 [B, C/4, T, V]
        gconv_parts = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)  # 按 head 拆分
        gconv_out_parts = []
        for i in range(self.gconv.shape[0]):
            # 使用 einsum 实现图卷积 ('n c t u, v u -> n c t v')
            z = torch.einsum('n c t u, v u -> n c t v', gconv_parts[i], self.gconv[i])
            gconv_out_parts.append(z)
        y_gconv = torch.cat(gconv_out_parts, dim=1)  # [B, (C/4)*(num_heads//4), T, V]
        y_gconv = self.proj_gconv(y_gconv)  # 投影到 in_channels

        # T-Conv 分支
        y_tconv = self.tconv(split_f_conv[1])  # [B, C/4, T, V]
        y_tconv = self.proj_tconv(y_tconv)  # 投影到 in_channels

        # 注意力分支（Skate-MSA）
        attn_branch_outputs = []
        split_f_attn = torch.chunk(f_attn, len(self.partition_function), dim=1)  # 分为多个注意力子分支
        for i in range(len(self.partition_function)):
            branch = split_f_attn[i]  # [B, C_attn, T, V]，其中 C_attn = in_channels//(len(partition)*2)
            partitioned = self.partition_function[i](branch, self.partition_size[i])  # 分块
            B_new, _, _, C_branch = partitioned.shape
            partitioned = partitioned.view(B_new, self.partition_size[i][0] * self.partition_size[i][1], C_branch)
            attn_out = self.attention[i](partitioned)  # 注意力计算
            attn_out = self.reverse_function[i](attn_out, (T, V), self.partition_size[i])  # 逆分块
            attn_branch_outputs.append(attn_out)
        # 融合多个注意力分支
        y_attn = self.attn_fusion_module(attn_branch_outputs)
        y_attn = self.proj_attn(y_attn)  # 投影到 in_channels

        # 融合所有三个主分支
        branches = [y_gconv, y_tconv, y_attn]
        fused = self.fusion_module(branches, skip)  # 融合并加入跳跃连接

        # 最终投影及残差连接
        fused_proj = self.proj(fused.permute(0, 2, 3, 1).contiguous())
        fused_proj = self.proj_drop(fused_proj)
        fused_proj = fused_proj.permute(0, 3, 1, 2).contiguous()  # 转回 [B, C, T, V]
        out = skip + self.drop_path(fused_proj)

        # Feed-Forward MLP
        residual = out  # 保存原始输入，形状 [B, C, T, V]
        out = out.permute(0, 2, 3, 1).contiguous()  # 转换为 [B, T, V, C]
        out = self.norm_2(out)  # 在最后一个维度归一化
        out = self.mlp(out)  # MLP 处理
        out = self.drop_path(out)
        out = residual.permute(0, 2, 3, 1).contiguous() + out  # 加上残差
        out = out.permute(0, 3, 1, 2).contiguous()  # 再转换回 [B, C, T, V]
        return out


""" 以下模块基本保持不变 """


class PatchMergingTconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1),
                                   dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.bn(self.reduction(x))
        return x


class SkateFormerBlockDS(nn.Module):
    def __init__(
            self, in_channels, out_channels, num_points=50, kernel_size=7, downscale=False, num_heads=32,
            type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
            attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
            act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm):
        super(SkateFormerBlockDS, self).__init__()

        if downscale:
            self.downsample = PatchMergingTconv(in_channels, out_channels, kernel_size=kernel_size)
        else:
            self.downsample = None

        self.transformer = SkateFormerBlock(
            in_channels=out_channels,
            num_points=num_points,
            kernel_size=kernel_size,
            num_heads=num_heads,
            type_1_size=type_1_size,
            type_2_size=type_2_size,
            type_3_size=type_3_size,
            type_4_size=type_4_size,
            attn_drop=attn_drop,
            drop=drop,
            rel=rel,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer,
        )

    def forward(self, input):
        if self.downsample is not None:
            output = self.transformer(self.downsample(input))
        else:
            output = self.transformer(input)
        return output


class SkateFormerStage(nn.Module):
    def __init__(
            self, depth, in_channels, out_channels, first_depth=False,
            num_points=50, kernel_size=7, num_heads=32,
            type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
            attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
            act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm):
        super(SkateFormerStage, self).__init__()
        blocks = []
        for index in range(depth):
            blocks.append(
                SkateFormerBlockDS(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    num_points=num_points,
                    kernel_size=kernel_size,
                    downscale=((index == 0) and (not first_depth)),
                    num_heads=num_heads,
                    type_1_size=type_1_size,
                    type_2_size=type_2_size,
                    type_3_size=type_3_size,
                    type_4_size=type_4_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    rel=rel,
                    drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        output = input
        for block in self.blocks:
            output = block(output)
        return output


class SkateFormer(nn.Module):
    def __init__(self, in_channels=3, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192), num_classes=60,
                 coarse_classes=7,
                 embed_dim=64, num_people=2, num_frames=64, num_points=50, kernel_size=7, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., head_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm, index_t=False, global_pool='avg',
                 projection_dim=128):

        super(SkateFormer, self).__init__()

        assert len(depths) == len(channels), "每个阶段必须给定通道数 (For each stage a channel dimension must be given)."
        assert global_pool in ["avg", "max"], f"仅支持 avg 和 max, 但给定了 {global_pool}"
        self.num_classes = num_classes
        self.head_drop = head_drop
        self.index_t = index_t
        self.embed_dim = embed_dim

        if self.head_drop != 0:
            self.dropout = nn.Dropout(p=self.head_drop)
        else:
            self.dropout = None

        # Stem 网络 (Stem Network)
        stem = []
        stem.append(nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0)))
        stem.append(act_layer())
        stem.append(
            nn.Conv2d(in_channels=2 * in_channels, out_channels=3 * in_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)))
        stem.append(act_layer())
        stem.append(nn.Conv2d(in_channels=3 * in_channels, out_channels=embed_dim, kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0)))
        self.stem = nn.ModuleList(stem)

        if self.index_t:
            self.joint_person_embedding = nn.Parameter(torch.zeros(embed_dim, num_points * num_people))
            trunc_normal_(self.joint_person_embedding, std=.02)
        else:
            self.joint_person_temporal_embedding = nn.Parameter(
                torch.zeros(1, embed_dim, num_frames, num_points * num_people))
            trunc_normal_(self.joint_person_temporal_embedding, std=.02)

        # 初始化阶段 (Initialize Stages)
        drop_path_vals = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            stages.append(
                SkateFormerStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    first_depth=index == 0,
                    num_points=num_points * num_people,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    type_1_size=type_1_size,
                    type_2_size=type_2_size,
                    type_3_size=type_3_size,
                    type_4_size=type_4_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    rel=rel,
                    drop_path=drop_path_vals[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer
                )
            )
        self.stages = nn.ModuleList(stages)
        self.global_pool = global_pool

        self.head = nn.Linear(channels[-1], num_classes)
        self.coarse_head = nn.Linear(channels[-1], coarse_classes)

        self.projection_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[-1], projection_dim)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, input):
        output = input
        for stage in self.stages:
            output = stage(output)
        return output

    def forward_head(self, input, pre_logits=False):
        if self.global_pool == "avg":
            input = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            input = torch.amax(input, dim=(2, 3))
        if self.dropout is not None:
            input = self.dropout(input)
        return input if pre_logits else self.head(input), input, self.coarse_head(input)

    def forward_data(self, input, index_t):
        B, C, T, V, M = input.shape
        output = input.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  # [B, C, T, M * V]
        for layer in self.stem:
            output = layer(output)
        if self.index_t:
            te = torch.zeros(B, T, self.embed_dim).to(output.device)
            div_term = torch.exp(
                (torch.arange(0, self.embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.embed_dim))).to(
                output.device)
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            output = output + torch.einsum('b t c, c v -> b c t v', te, self.joint_person_embedding)
        else:
            output = output + self.joint_person_temporal_embedding
        return output

    def forward(self, input, index_t):
        output = self.forward_data(input, index_t)
        output = self.forward_features(output)
        output_forward_head, output_pooling, output_coarse_head = self.forward_head(output)
        output_projection_head = self.projection_head(output_pooling)
        return output_forward_head, output_projection_head, output_coarse_head


def SkateFormer_(**kwargs):
    return SkateFormer(
        depths=(2, 2, 2, 2),
        channels=(96, 192, 192, 192),
        embed_dim=96,
        **kwargs
    )
