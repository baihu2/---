import torch
import numpy as np

def ctc_decode(preds, char_dict, blank_index=0):
    """
    CTC 贪心解码
    
    Args:
        preds: [T, B, num_classes] - 模型输出
        char_dict: {index: character} 映射字典
        blank_index: blank token 的索引 (默认0)
    
    Returns:
        decoded_texts: 解码后的文本列表
    """
    # 获取预测的类别索引 [T, B]
    _, pred_indices = preds.max(2)  # [T, B]
    pred_indices = pred_indices.cpu().numpy()
    
    decoded_texts = []
    for b in range(pred_indices.shape[1]):
        # 获取当前样本的预测序列
        seq = pred_indices[:, b]
        
        # CTC 解码: 合并重复 + 移除 blank
        decoded = []
        prev_char = None
        for idx in seq:
            if idx != blank_index and idx != prev_char:
                decoded.append(char_dict.get(idx, ''))
            prev_char = idx
        
        decoded_text = ''.join(decoded)
        decoded_texts.append(decoded_text)
    
    return decoded_texts