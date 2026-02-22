# -*- coding: utf-8 -*-
"""
CRNN 车牌识别训练脚本 - CBLPRD-330k 专用 (GPU 优化 + 断点续训 + 收敛加速)
【关键优化】
1. 优化学习率策略：ReduceLROnPlateau + 学习率预热
2. 增加训练轮数：EPOCHS=30（35万图片推荐）
3. 数据增强：随机水平翻转（车牌方向无关）
4. 损失监控：验证集损失作为学习率调整依据
5. 优化训练循环：更有效的进度显示
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import random

# ----------------------------
# 路径配置：定义项目根目录、图像文件夹、训练/验证列表及检查点路径
# ----------------------------
ROOT_DIR = os.getcwd()  # 获取当前工作目录
IMG_DIR = os.path.join(ROOT_DIR, "CBLPRD-330k")  # 图像数据集路径
TRAIN_TXT = os.path.join(ROOT_DIR, "train.txt")   # 训练集标签文件
VAL_TXT = os.path.join(ROOT_DIR, "val.txt")       # 验证集标签文件
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoint.pth")  # 断点续训保存路径

print(f"📁 项目目录: {ROOT_DIR}")
# 断言检查：确保关键路径存在，否则报错退出
assert os.path.exists(IMG_DIR), f"❌ 图像文件夹不存在: {IMG_DIR}"
assert os.path.exists(TRAIN_TXT), f"❌ train.txt 不存在: {TRAIN_TXT}"
assert os.path.exists(VAL_TXT), f"❌ val.txt 不存在: {VAL_TXT}"

# ----------------------------
# 字符集定义：支持中国车牌所有合法字符（省份+字母+数字+特殊标识）
# ----------------------------
PROVINCES = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
    '港', '澳', '挂', '学', '领', '使', '临'
]
LETTERS = [chr(ord('A') + i) for i in range(26)]  # A-Z
DIGITS = [str(i) for i in range(10)]              # 0-9

# 构建完整字符表：索引0为CTC的blank标记
CHARS = ['<blank>'] + PROVINCES + LETTERS + DIGITS
CHAR2IDX = {ch: idx for idx, ch in enumerate(CHARS)}  # 字符 → 索引
IDX2CHAR = {idx: ch for ch, idx in CHAR2IDX.items()}  # 索引 → 字符
NUM_CLASSES = len(CHARS)

print(f"🔤 字符集大小: {NUM_CLASSES} (含 blank)")

# ----------------------------
# 超参数配置：根据GPU显存和数据规模优化
# ----------------------------
BATCH_SIZE = 32        # 批次大小（显存允许下尽量大）
EPOCHS = 30            # 总训练轮数（大数据集需更多epoch）
LEARNING_RATE = 0.0005 # 初始学习率（较小值利于收敛）
IMG_HEIGHT = 32        # 输入图像高度（CRNN要求高度固定）
IMG_WIDTH = 280        # 输入图像宽度（足够容纳7位车牌）
LOG_INTERVAL = 1000    # 每隔多少batch打印日志
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {DEVICE} | BATCH_SIZE={BATCH_SIZE} | EPOCHS={EPOCHS}")

# ----------------------------
# CRNN 模型定义：CNN + BiLSTM + FC，专为不定长文本识别设计
# ----------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes, imgH=32, nc=1, nh=256):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH必须是16的倍数（因CNN下采样）'
        
        # CNN 特征提取器：逐步下采样，最终高度压缩为1
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)  # 最终输出 [B, 512, 1, W']
        )
        # 双向LSTM：捕捉字符上下文依赖
        self.rnn = nn.LSTM(512, nh, num_layers=2, bidirectional=True, batch_first=True)
        # 全连接层：映射到字符类别空间
        self.fc = nn.Linear(nh * 2, num_classes)
        
        # 权重初始化：使用He/Xavier提升收敛速度
        self._initialize_weights()

    def _initialize_weights(self):
        """对卷积层、线性层、LSTM进行合理初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        """前向传播：输入 [B, 1, H, W] → 输出 [B, T, num_classes]"""
        conv = self.cnn(x)  # [B, 512, 1, W']
        b, c, h, w = conv.size()
        assert h == 1, "CNN输出高度必须为1"
        rnn_input = conv.squeeze(2).permute(0, 2, 1)  # [B, W', 512]
        rnn_out, _ = self.rnn(rnn_input)  # [B, W', 512]
        output = self.fc(rnn_out)  # [B, W', num_classes]
        return output

# ----------------------------
# 自定义数据集类：支持训练/验证模式、数据清洗、增强
# ----------------------------
class LicensePlateDataset(Dataset):
    def __init__(self, txt_path, img_dir, debug=False, is_train=True):
        self.img_dir = img_dir
        self.debug = debug
        self.is_train = is_train  # 标记是否为训练集（决定是否增强）
        self.data = []
        # 读取标签文件：每行格式 "filename.jpg label"
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path_in_txt = parts[0]
                    label = parts[1]
                    filename = os.path.basename(img_path_in_txt)
                    full_path = os.path.join(img_dir, filename)
                    if os.path.exists(full_path):
                        self.data.append((filename, label))
                    elif debug:
                        print(f"⚠️ 图像不存在: {full_path}")
        print(f"✅ 加载 {len(self.data)} 条有效样本 from {txt_path} (训练集={self.is_train})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度图节省显存
        if image is None:
            image = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        else:
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # 统一尺寸
        
        # 清洗标签：只保留合法字符，避免无效标签导致崩溃
        valid_chars = set(CHARS[1:])  # 排除 <blank>
        cleaned_label = ''.join(ch for ch in label if ch in valid_chars)
        if len(cleaned_label) == 0:
            cleaned_label = "京A00000"  # 默认兜底
        
        # 将字符转为索引序列
        label_indices = [CHAR2IDX[ch] for ch in cleaned_label]
        
        # 转为PyTorch张量并归一化到 [-1, 1]
        image = torch.from_numpy(image).float().unsqueeze(0)  # [1, H, W]
        image = image / 255.0
        image = (image - 0.5) / 0.5  # 标准化
        
        # ✅ 数据增强：仅训练集启用随机水平翻转（车牌对称）
        if self.is_train and random.random() > 0.5:
            image = torch.flip(image, [2])  # 沿宽度维度翻转
        
        return image, label_indices, cleaned_label

# ----------------------------
# 批次合并函数：将变长标签序列展平，供CTCLoss使用
# ----------------------------
def collate_fn(batch):
    """
    输入: [(img, label_indices, label_str), ...]
    输出: 
        images: [B, 1, H, W]
        targets: [sum(label_lengths)] 所有标签拼接成一维
        target_lengths: [B] 每个样本的标签长度
        labels: [B] 原始字符串标签（用于计算准确率）
    """
    images, label_indices_list, labels = zip(*batch)
    images = torch.stack(images, 0)  # 合并图像
    
    flat_targets = []
    target_lengths = []
    for indices in label_indices_list:
        flat_targets.extend(indices)
        target_lengths.append(len(indices))
    
    targets = torch.LongTensor(flat_targets)
    target_lengths = torch.IntTensor(target_lengths)
    return images, targets, target_lengths, labels

# ----------------------------
# CTC解码函数：将模型输出转换为可读字符串（去除blank和重复）
# ----------------------------
def decode_ctc(outputs, output_lengths=None):
    """
    outputs: [B, T, num_classes] 模型原始输出
    output_lengths: [B] 每个样本的有效时间步（可选）
    返回: 解码后的字符串列表
    """
    _, preds = outputs.max(2)  # [B, T]
    preds = preds.cpu().numpy()
    
    decoded_strings = []
    B = preds.shape[0]
    
    if output_lengths is None:
        seq_lengths = [preds.shape[1]] * B
    else:
        output_lengths = output_lengths.cpu().numpy()
        if len(output_lengths) != B:
            # 处理长度不匹配（如最后一个batch不足）
            seq_lengths = output_lengths[:B] if len(output_lengths) > B else np.concatenate([output_lengths, [preds.shape[1]] * (B - len(output_lengths))])
        else:
            seq_lengths = output_lengths

    for i in range(B):
        length = int(seq_lengths[i])
        seq = preds[i][:length]
        out = []
        prev = -1
        for p in seq:
            if p != prev and p != 0:  # 跳过blank(0)和连续重复
                out.append(IDX2CHAR[p])
            prev = p
        decoded_strings.append(''.join(out))
    return decoded_strings

# ----------------------------
# 检查点保存与加载：支持断点续训
# ----------------------------
def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, train_losses, val_losses, val_accuracies, path):
    """保存训练状态，便于恢复"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    torch.save(checkpoint, path)
    print(f"💾 检查点已保存至: {path}")

def load_checkpoint(path, model, optimizer, scheduler):
    """从检查点恢复训练"""
    if not os.path.exists(path):
        print("🔍 未找到检查点，从头开始训练")
        return 0, 0.0, [], [], []
    
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ 已从 {path} 恢复训练，继续从 Epoch {checkpoint['epoch'] + 1} 开始")
    return (
        checkpoint['epoch'] + 1,
        checkpoint['best_acc'],
        checkpoint['train_losses'],
        checkpoint['val_losses'],
        checkpoint['val_accuracies']
    )

# ----------------------------
# 主训练函数：包含学习率调度、验证、保存最佳模型等
# ----------------------------
def train(resume=True):
    print("🔍 加载训练集...")
    train_dataset = LicensePlateDataset(TRAIN_TXT, IMG_DIR, debug=True, is_train=True)
    print("🔍 加载验证集...")
    val_dataset = LicensePlateDataset(VAL_TXT, IMG_DIR, debug=False, is_train=False)

    # 创建DataLoader：启用多进程加速数据加载
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,      # 并行加载数据
        pin_memory=True,    # GPU加速
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 初始化模型、损失函数、优化器
    model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)  # CTC损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ✅ 学习率调度：先预热5轮，再根据验证损失动态调整
    warmup_epochs = 5
    scheduler_warmup = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 初始化训练状态
    start_epoch = 0
    best_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 尝试加载检查点
    if resume and os.path.exists(CHECKPOINT_PATH):
        start_epoch, best_acc, train_losses, val_losses, val_accuracies = load_checkpoint(
            CHECKPOINT_PATH, model, optimizer, scheduler_plateau
        )

    total_train_batches = len(train_loader)
    print(f"\n🚀 开始训练... 总共 {total_train_batches} batches/epoch | EPOCHS={EPOCHS}\n")

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        batch_correct = 0
        batch_total = 0
        
        # 训练循环
        for i, (images, targets, target_lengths, labels) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            target_lengths = target_lengths.to(DEVICE, non_blocking=True)

            # 前向传播
            outputs = model(images)  # [B, T, num_classes]
            output_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long)
            outputs_logprob = outputs.log_softmax(2).permute(1, 0, 2)  # [T, B, C] for CTC
            loss = criterion(outputs_logprob, targets, output_lengths, target_lengths)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 实时计算训练准确率（用于日志）
            with torch.no_grad():
                preds = decode_ctc(outputs)
                for pred, gt in zip(preds, labels):
                    if pred == gt:
                        batch_correct += 1
                    batch_total += 1

            # 日志输出
            if (i + 1) % LOG_INTERVAL == 0 or i == len(train_loader) - 1:
                avg_loss = total_loss / (i + 1)
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i+1}/{total_train_batches} | "
                      f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                      f"Batch Acc: {batch_acc:.4f} ({batch_correct}/{batch_total})")
        
        # 计算平均训练损失
        avg_train_loss = total_loss / total_train_batches
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets, target_lengths, labels in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                output_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long)
                outputs_logprob = outputs.log_softmax(2).permute(1, 0, 2)
                loss = criterion(outputs_logprob, targets, output_lengths, target_lengths)
                val_loss += loss.item()
                
                preds = decode_ctc(outputs)
                for pred, gt in zip(preds, labels):
                    if pred == gt:
                        correct += 1
                    total += 1

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        # 更新学习率：预热期后切换为Plateau调度
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_plateau.step(val_loss)  # 关键：用验证损失调整
        
        # 打印epoch总结
        epoch_time = time.time() - epoch_start
        print(f"\n✅ Epoch {epoch+1} 完成 | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # 保存最佳模型（按验证准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ROOT_DIR, "crnn_best.pth"))
            print(f"🎉 保存最佳模型! Acc: {best_acc:.4f} (Val Loss: {val_loss:.4f})\n")

        # 保存完整检查点（用于断点续训）
        save_checkpoint(
            epoch, model, optimizer, scheduler_plateau, best_acc, 
            train_losses, val_losses, val_accuracies, CHECKPOINT_PATH
        )

    # 绘制训练曲线并保存
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.title('Training & Validation Loss'); plt.xlabel('Epoch'); plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='green')
    plt.title('Validation Accuracy'); plt.xlabel('Epoch'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "training_curves.png"))
    plt.show()

    print(f"\n🎯 训练完成！最佳准确率: {best_acc:.4f} | 最佳验证损失: {val_loss:.4f}")
    return model

# ----------------------------
# 测试函数：评估最终模型性能
# ----------------------------
def test(model, val_loader, device):
    print("\n🔍 开始最终测试评估...")
    model.eval()
    correct = 0
    total = 0
    examples = []

    with torch.no_grad():
        for images, targets, target_lengths, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = decode_ctc(outputs)
            for pred, gt in zip(preds, labels):
                if pred == gt:
                    correct += 1
                total += 1
                if len(examples) < 5:
                    examples.append((gt, pred))

    test_acc = correct / total
    print(f"\n✅ 最终测试准确率: {test_acc:.4f} ({correct}/{total})")

    print("\n📊 预测样例（真实 → 预测）:")
    for gt, pred in examples:
        status = "✅" if gt == pred else "❌"
        print(f"  {status} {gt} → {pred}")

    return test_acc

# ----------------------------
# 主程序入口
# ----------------------------
if __name__ == "__main__":
    trained_model = train(resume=True)

    print("\n📥 加载最佳模型进行最终测试...")
    best_model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    best_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "crnn_best.pth"), map_location=DEVICE))
    
    val_dataset_final = LicensePlateDataset(VAL_TXT, IMG_DIR, debug=False, is_train=False)
    val_loader_final = DataLoader(
        val_dataset_final,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    final_acc = test(best_model, val_loader_final, DEVICE)