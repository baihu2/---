
import os
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO

# ========== 配置参数 ==========
# 明确指定 CPU（因为 GPU 不可用）
DEVICE = torch.device('cpu')
print(f"使用设备: {DEVICE}")

# 路径配置（修正为预处理脚本实际生成的路径）
BASE_PATH = r"C:\Users\白狐\Desktop\新建文件夹\BiYeSheJi\CCPD2020\CCPD2020\CCPD2020_green_processed"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train", "images")  # ✅ 修正为 "images"
VAL_IMG_DIR = os.path.join(BASE_PATH, "val", "images")      # ✅ 修正为 "images"

def verify_dataset():
    """验证数据集是否完整"""
    print("🔍 验证数据集完整性...")
    
    # 检查 images 和 labels 是否匹配
    train_img_dir = Path(TRAIN_IMG_DIR)
    train_label_dir = Path(BASE_PATH) / "train" / "labels"
    
    if not train_img_dir.exists():
        print(f"❌ 训练图片目录不存在: {train_img_dir}")
        return False
    
    if not train_label_dir.exists():
        print(f"❌ 训练标签目录不存在: {train_label_dir}")
        return False
    
    img_files = set(f.stem for f in train_img_dir.glob("*.jpg"))
    label_files = set(f.stem for f in train_label_dir.glob("*.txt"))
    
    common_files = img_files & label_files
    print(f"训练集: 图片={len(img_files)}, 标签={len(label_files)}, 匹配={len(common_files)}")
    
    if len(common_files) == 0:
        print("❌ 警告: 没有匹配的图片和标签文件！")
        return False
    
    # 验证第一个样本
    sample_stem = next(iter(common_files))
    sample_img = train_img_dir / f"{sample_stem}.jpg"
    sample_label = train_label_dir / f"{sample_stem}.txt"
    
    # 检查图片可读取（支持中文路径）
    img_array = np.fromfile(str(sample_img), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ 样本图片无法读取: {sample_img}")
        return False
    
    # 检查标签格式
    try:
        with open(sample_label, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            if len(parts) != 5:
                print(f"❌ 标签格式错误: {sample_label}")
                return False
            
            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            
            # 验证坐标范围
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print(f"❌ 标签坐标超出范围: {sample_label}")
                return False
                
    except Exception as e:
        print(f"❌ 标签读取错误 {sample_label}: {e}")
        return False
    
    print("✅ 数据集验证通过！")
    return True

def train_yolo_detector():
    """训练 YOLOv8 车牌检测模型（CPU 优化版）"""
    print("\n🚀 开始训练 YOLOv8 车牌检测模型...")
    
    # 加载预训练模型 (nano 版本，适合笔记本)
    model = YOLO('yolov8n.pt')
    
    # 训练配置 (CPU 友好)
    results = model.train(
        data=os.path.join(BASE_PATH, 'data.yaml'),  # 使用完整路径
        epochs=15,             # 适当减少轮数
        batch=4,               # CPU 必须减小 batch size
        imgsz=416,             # 降低输入尺寸（更适合竖屏图像）
        patience=5,            # 内置早停: 5轮无改善停止
        device='cpu',          # 明确指定 CPU
        close_mosaic=0,        # 禁用 mosaic 增强（解决大目标问题）
        augment=False,         # 禁用数据增强（CPU 更稳定）
        workers=2,             # 减少数据加载线程（避免 Windows 问题）
        cache=False,           # 禁用缓存（节省内存）
        name='plate_detection' # 保存目录名
    )
    
    print("✅ YOLOv8 训练完成!")
    return model

# ========== 主函数 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("车牌检测与识别系统 - YOLOv8 训练脚本 (CPU 优化版)")
    print("=" * 60)
    
    # 验证数据集
    if not verify_dataset():
        print("❌ 数据集验证失败，退出训练")
        exit(1)
    
    # 训练 YOLOv8 检测器
    try:
        yolo_model = train_yolo_detector()
        print("\n🎉 YOLOv8 训练成功完成！")
        print(f"模型保存位置: runs/detect/plate_detection/weights/best.pt")
    except Exception as e:
        print(f"❌ YOLOv8 训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n💡 后续步骤:")
    print("1. 使用训练好的 YOLOv8 模型裁剪生成车牌图像数据集")
    print("2. 用裁剪后的图像训练 CRNN 字符识别模型")
    print("3. 集成两个模型进行端到端测试")