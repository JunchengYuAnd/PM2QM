import pandas as pd
from pathlib import Path
import input_midi_process as im
import label_midi_process as lm
import batch_sep as bs
import torch

DATASET_FOLDER = Path("asap-dataset")
csv_path = "asap-dataset/metadata.csv"
df = pd.read_csv(csv_path)

all_batches = []

p_midi_paths = df['midi_performance'].apply(lambda x: DATASET_FOLDER / x)
q_midi_paths = df['midi_score'].apply(lambda x: DATASET_FOLDER / x)

for p_midi, q_midi in zip(p_midi_paths, q_midi_paths):
    if not p_midi.exists():
        print(f"⚠ Warning: 文件 {p_midi} 不存在！")
        continue
    if not q_midi.exists():
        print(f"⚠ Warning: 文件 {q_midi} 不存在！")
        continue

    input_tokens = im.midi_to_tokens(str(p_midi))
    label_tokens = lm.midi_to_tokens(str(q_midi))

    if not input_tokens or not label_tokens:
        print(f"⚠ Warning: {p_midi} 或 {q_midi} 的 tokens 为空，跳过处理！")
        continue

    # 调用分段函数
    input_batches, input_masks, label_batches, label_masks = \
        bs.proportional_batch_split(input_tokens, label_tokens)

    if input_batches.numel() == 0 or label_batches.numel() == 0:
        print(f"⚠ Warning: {p_midi} 或 {q_midi} 生成的 batch 为空，跳过存储！")
        continue

    # ======= 打印各种统计信息 =======
    # 1) 输入序列的形状、列最大值
    # 因为 shape 是 (batch_size, seq_len, 4)，先 reshape 成 (N, 4)
    ib_reshaped = input_batches.reshape(-1, 4)
    max_input_per_col = ib_reshaped.max(dim=0)[0]  # 对每一列做最大值
    print("文件:", p_midi.name, "→ input_batches shape:", input_batches.shape)
    print("  => input_batches每列最大值 (pitch, onset, duration, velocity):",
          max_input_per_col.tolist())  # 打印成列表

    # 2) 标签序列
    lb_reshaped = label_batches.reshape(-1, 4)
    max_label_per_col = lb_reshaped.max(dim=0)[0]
    print("文件:", q_midi.name, "→ label_batches shape:", label_batches.shape)
    print("  => label_batches每列最大值 (pitch, onset, duration, velocity):",
          max_label_per_col.tolist())

    # 3) Mask 也可以查看一下 unique 值
    print("  => input_masks shape:", input_masks.shape,
          "unique:", input_masks.unique().tolist())
    print("  => label_masks shape:", label_masks.shape,
          "unique:", label_masks.unique().tolist())

    # 将结果保存到列表
    all_batches.append({
        "input_batches": input_batches,
        "input_masks": input_masks,
        "label_batches": label_batches,
        "label_masks": label_masks
    })

# 循环结束后，再统一保存
torch.save(all_batches, "midi_batches.pt")
print("✅ 批量 MIDI 预处理完成，数据已保存！")
