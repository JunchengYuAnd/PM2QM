import input_midi_process as im
import label_midi_process as lm
import numpy as np
import torch
import math



def pad_and_mask_batches(batches, max_len=128, pad_token=0):
    """
    将 batch 填充到固定长度，并生成 Mask 矩阵
    参数：
        batches (list): 需要填充的 batch 列表
        max_len (int): 目标长度 (默认 128)
        pad_token (int): 填充的 Token 值 (默认 0)

    返回：
        padded_batches (torch.Tensor): 填充后的 batch (形状: batch_size x max_len)
        attention_masks (torch.Tensor): Mask 矩阵 (形状: batch_size x max_len)
    """
    batch_size = len(batches)
    padded_batches = np.full((batch_size, max_len, 4), pad_token, dtype=np.int64)  # 初始化填充张量
    attention_masks = np.zeros((batch_size, max_len, 4), dtype=np.int64)  # 初始化 Mask

    for i, batch in enumerate(batches):
        seq_len = min(len(batch), max_len)  # 确保不超过 128
        padded_batches[i, :seq_len] = batch[:seq_len]  # 填充实际值
        attention_masks[i, :seq_len] = 1  # 真实 token 位置填充 1

    return torch.tensor(padded_batches), torch.tensor(attention_masks)


def proportional_batch_split(input_tokens, label_tokens,
                             max_batch_size=128, overlap=64, pad_token=0):
    """
    将 input_tokens 和 label_tokens 做按比例的分段切分并进行填充。
    返回: (input_batches, input_masks, label_batches, label_masks)
          每个 shape = [batch_size, max_batch_size, 4]
    """
    input_tokens = np.array(input_tokens, dtype=object)
    label_tokens = np.array(label_tokens, dtype=object)

    # 1) 判断长短序列
    input_total = len(input_tokens)
    label_total = len(label_tokens)
    if input_total == 0 or label_total == 0:
        # 若有空序列可自行决定如何处理
        raise ValueError("输入或标签序列为空，无法分割。")

    if input_total > label_total:
        long_tokens, short_tokens = input_tokens, label_tokens
    else:
        long_tokens, short_tokens = label_tokens, input_tokens

    long_total = len(long_tokens)
    short_total = len(short_tokens)

    # 2) 计算批次数量
    # 避免 n_batches 过多或只有 1
    n_batches = math.ceil(long_total / max_batch_size)
    if n_batches == 0:
        n_batches = 1
    batch_size_long = max_batch_size
    batch_size_short = int(round(short_total / n_batches))
    print(batch_size_short,batch_size_long)

    # 3) 计算 overlap 对短序列的分块
    # 保持与长序列相同或按比例?
    # 这里示例: 同一个 overlap
    # short_overlap = ...
    short_overlap = overlap

    # 4) 实现函数分段
    def split_with_overlap(arr, chunk_size, ov):
        """将 arr 分成多个 [chunk_size] 长度的段，每段与下一段重叠 ov 大小"""
        results = []
        i = 0
        while i < len(arr):
            end_index = i + chunk_size
            chunk = arr[i:end_index]
            if len(chunk) == 0:
                break
            results.append(chunk)
            i += (chunk_size - ov)  # 下一段起点前移
            if i < 0:  # 理论上防止负数
                break
        return results

    long_batches = split_with_overlap(long_tokens, batch_size_long, overlap)
    short_batches = split_with_overlap(short_tokens, batch_size_short, short_overlap)

    # 5) onset 归一化
    def normalize_onset(batches_list):
        for i in range(len(batches_list)):
            batch = batches_list[i]
            if len(batch) > 0:
                base_onset = batch[0][1]
                # 把每个 token 的 onset 减去 base_onset
                new_batch = []
                for p, o, d, v in batch:
                    new_batch.append((p, o - base_onset, d, v))
                batches_list[i] = new_batch

    normalize_onset(long_batches)
    normalize_onset(short_batches)

    # 6) 保证两者 batch 数量匹配
    min_batches = min(len(long_batches), len(short_batches))
    long_batches = long_batches[:min_batches]
    short_batches = short_batches[:min_batches]

    # 7) 组合为 input / label
    #   如果原本 input更长 => input_batches = long_batches
    #   否则 => input_batches = short_batches
    if input_total > label_total:
        input_batches = long_batches
        label_batches = short_batches
    else:
        input_batches = short_batches
        label_batches = long_batches

    # 8) 填充和生成 mask
    input_batches, input_masks = pad_and_mask_batches(input_batches, max_len=max_batch_size, pad_token=pad_token)
    label_batches, label_masks = pad_and_mask_batches(label_batches, max_len=max_batch_size, pad_token=pad_token)

    return input_batches, input_masks, label_batches, label_masks




if __name__ == "__main__":
    input_path = "asap-dataset/Rachmaninoff/Preludes_op_32/5/KorchinskayaKogan03.mid"
    label_path = "asap-dataset/Rachmaninoff/Preludes_op_32/5/midi_score.mid"
    input_token = im.midi_to_tokens(input_path)
    label_token = lm.midi_to_tokens(label_path)
    input_batches, input_masks, label_batches, label_masks= proportional_batch_split(input_token, label_token)
    print(input_batches.shape, input_masks.shape, label_batches, label_masks.shape)

