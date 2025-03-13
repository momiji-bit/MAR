import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from main_contrast import *

# 假设 device、arg、processor 等已经初始化
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = get_parser()
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

# ----------------------------
# 1. 加载对比学习预训练模型的检查点 (Load Pre-trained Contrastive Model Checkpoint)
# ----------------------------
# 指定检查点路径（请根据实际情况修改）
checkpoint_path = 'contrastive_model_epoch17.pth'
# 构造骨架模型 (SkateFormer) 和对比学习模型包装器 (Contrastive Model Wrapper)
backbone = SkateFormer_(**arg.model_args).to(device)
contrastive_model = ContrastiveSkateFormer(backbone, projection_dim=128).to(device)
# 加载保存的检查点模型参数
checkpoint = torch.load(checkpoint_path, map_location=device)
contrastive_model.load_state_dict(checkpoint)
print("成功加载检查点模型：", checkpoint_path)

# ----------------------------
# 2. 提取预训练的骨干网络，并冻结其参数 (Extract and Freeze Backbone)
# ----------------------------
# 提取对比学习模型中的骨干网络 (backbone)
backbone = contrastive_model.backbone
# 冻结骨干网络的所有参数，防止在分类 head 训练时更新
for param in backbone.parameters():
    param.requires_grad = False

# ----------------------------
# 3. 定义新的分类 head (Define New Classification Head)
# ----------------------------
# 假设你的数据集类别数为 num_classes（请根据实际情况修改）
num_classes = 52  # 示例：60 个类别 (60 classes)
# 获取骨干网络输出特征维度 (feature dimension)，这里使用 backbone.head.in_features 或 backbone.embed_dim
if hasattr(backbone, "head") and hasattr(backbone.head, "in_features"):
    in_features = backbone.head.in_features
else:
    in_features = backbone.embed_dim

# 定义一个简单的线性分类器 (Linear Classifier)
classifier = nn.Linear(in_features, num_classes).to(device)

# ----------------------------
# 4. 定义优化器和损失函数 (Define Optimizer and Loss Function)
# ----------------------------
optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
criterion_cls = nn.CrossEntropyLoss()  # 交叉熵损失 (Cross Entropy Loss)

# ----------------------------
# 5. 训练分类 head (Train the Classification Head)
# ----------------------------
num_epochs = 20  # 分类 head 的训练 epoch 数，根据需要调整
for epoch in range(num_epochs):
    classifier.train()
    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    pbar = tqdm(processor.data_loader['train'], desc=f"Classifier Training Epoch {epoch + 1}/{num_epochs}")
    for batch in pbar:
        # 假设数据加载器返回 (data, index_t, labels, _) 四项
        data, index_t, labels, _ = batch
        data = data.float().to(device)
        index_t = index_t.float().to(device)
        labels = labels.to(device)

        # 利用预训练的骨干网络提取特征 (Feature Extraction)
        with torch.no_grad():
            output = backbone.forward_data(data, index_t)
            features = backbone.forward_features(output)
            # forward_head 返回 pre_logits 特征 (提取全局特征)
            pooled_features = backbone.forward_head(features, pre_logits=True)

        # 前向传播：将提取到的特征输入到新的分类 head 中 (Forward pass through classifier)
        logits = classifier(pooled_features)
        loss = criterion_cls(logits, labels)

        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_cls.step()

        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
        # 计算 Top1 预测准确率 (Top1 Accuracy)
        _, preds = torch.max(logits, 1)
        correct_top1 += (preds == labels).sum().item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / total_samples
    acc = 100.0 * correct_top1 / total_samples
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Top1 Acc: {acc:.2f}%")

    # 可选：保存分类 head 的检查点 (Save classifier checkpoint)
    classifier_save_path = os.path.join(arg.work_dir, f'classifier_epoch{epoch + 1}.pth')
    torch.save(classifier.state_dict(), classifier_save_path)
    print("已保存分类 head 检查点：", classifier_save_path)
