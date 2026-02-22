import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class PlateRecognitionDataset(Dataset):
    """
    车牌字符识别数据集
    假设标签文件格式: 图像名.jpg 对应 标签.txt
    标签内容: "京A12345" (字符串)
    """
    def __init__(self, image_dir, label_dir, transform=None, char_to_idx=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform
        
        # 字符映射表 (70个字符: 34省+26字母+10数字)
        if char_to_idx is None:
            provinces = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                        '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                        '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                        '新', '港', '澳', '学']
            letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
            digits = [str(i) for i in range(10)]
            all_chars = ['<blank>'] + provinces + letters + digits  # CTC 需要 blank token
            self.char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        else:
            self.char_to_idx = char_to_idx
        
        self.num_classes = len(self.char_to_idx)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}, 使用空白图像")
            image = Image.new('RGB', (128, 32), (0, 0, 0))
        
        # 获取标签
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                label_text = f.read().strip()
        else:
            # 如果没有标签文件，从文件名解析 (CCPD 格式: 02599999_987777_014403_015501-128x48.png)
            # 这里简化处理，实际项目中需要完整的 CCPD 解析
            label_text = "京A12345"  # 默认标签
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label_text
    
    def text_to_tensor(self, text):
        """将文本转换为索引张量"""
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx['<blank>'])  # 未知字符用 blank 替代
        return torch.LongTensor(indices)

# 图像预处理
def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 128)),  # CRNN 标准输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])