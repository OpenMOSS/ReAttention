import torch

def quantize_along_penultimate_dim(tensor, n_bits):
    """
    沿着倒数第二维对四维张量进行量化
    :param tensor: 输入的四维张量 (batch_size, channel, height, width)
    :param n_bits: 量化位数
    :return: 量化后的张量，量化过程中的最小值和量化步长（用于还原）
    """
    # 计算量化的最大值和最小值
    min_val = tensor.min(dim=-2, keepdim=True)[0]  # 沿着倒数第二维求最小值
    max_val = tensor.max(dim=-2, keepdim=True)[0]  # 沿着倒数第二维求最大值

    # 量化范围
    range_val = max_val - min_val
    scale = range_val / (2**n_bits - 1)  # 量化步长

    # 将张量按倒数第二维量化
    quantized = torch.round((tensor - min_val) / scale) * scale + min_val

    return quantized, min_val, scale

def restore_from_quantized(quantized_tensor, min_val, scale):
    """
    从量化后的张量恢复数据
    :param quantized_tensor: 量化后的张量
    :param min_val: 量化过程中记录的最小值
    :param scale: 量化过程中记录的步长
    :return: 还原后的张量
    """
    # 恢复过程：首先反向计算
    restored_tensor = quantized_tensor * scale + min_val
    return restored_tensor