import torch


def analyze_pt_batches(file_path):
    """
    读取 .pt 文件并计算每个字段的最大值。

    参数：
        file_path (str): .pt 文件路径

    返回：
        max_values (dict): 每个字段的最大值
    """
    # 加载 .pt 文件
    data = torch.load(file_path)

    # 初始化存储最大值的字典
    max_values = {
        "input_batches": None,
        "input_masks": None,
        "label_batches": None,
        "label_masks": None
    }

    # 遍历批次数据
    for batch in data:
        for key in max_values.keys():
            if key in batch and isinstance(batch[key], torch.Tensor):
                if batch[key].numel() > 0:  # **确保 Tensor 不是空的**
                    batch_max = batch[key].max().item()
                    # 更新全局最大值
                    if max_values[key] is None or batch_max > max_values[key]:
                        max_values[key] = batch_max
                else:
                    print(f"⚠ Warning: {key} 为空 Tensor，跳过 max() 计算。")

    return max_values


# 运行代码
file_path = "midi_batches.pt"  # 请替换成你的 .pt 文件路径
max_values = analyze_pt_batches(file_path)

# 打印每个批次的最大值
for key, value in max_values.items():
    print(f"{key} 最大值: {value if value is not None else '无数据'}")
