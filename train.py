import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm

# 假设您的模型定义放在 model.py 中
from model import MidiTransformer

# ---------- 数据集 ------------
class MidiDataset(Dataset):
    """
    加载由 midi_batches.pt 存储的列表数据，每个元素是一个 dict:
      {
        "input_batches":  Tensor, shape [batch_size, seq_len, 4]
        "input_masks":    Tensor, shape [batch_size, seq_len] (或 [batch_size, seq_len, 4])
        "label_batches":  Tensor, shape [batch_size, seq_len, 4]
        "label_masks":    Tensor, shape [batch_size, seq_len]
      }
    注意: 这里 'batch_size' 指的是预处理时切分的内部维度,
          与 DataLoader 的 batch_size 并不冲突。
    """
    def __init__(self, data_path):
        super().__init__()
        self.all_data = torch.load(data_path)  # 这会是一个 list
        # self.all_data[i] 即一个 dict

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]  # 返回字典

# ---------- collate_fn ------------
def collate_fn(batch_list):
    """
    DataLoader 每次会给我们一个 batch_list，长度与 DataLoader(batch_size=?) 相同。
    但我们实际上希望一批只包含一个 dict(若每个 dict 本身就是一个batch)。
    如果您预处理时，每个 dict 就包含了大量 Token，可以令 DataLoader(batch_size=1)。

    如果您要把多个 dict 合并，需要在这里拼接 Tensors 并处理 mask。
    下方做了最简单的“单条合并”示例。
    """
    # 假设 batch_size=1, batch_list=[dict], 直接返回
    # 如果您想把这几个dict再拼成更大维度，请自行改写
    if len(batch_list) == 1:
        return batch_list[0]
    else:
        # 简单做一个示例：把 batch_list 中的 dict 分别展开后拼接
        # (仅供参考，可能不适合实际情况)
        combined = {
            "input_batches": [],
            "input_masks": [],
            "label_batches": [],
            "label_masks": []
        }
        for item in batch_list:
            combined["input_batches"].append(item["input_batches"])
            combined["input_masks"].append(item["input_masks"])
            combined["label_batches"].append(item["label_batches"])
            combined["label_masks"].append(item["label_masks"])

        combined["input_batches"] = torch.cat(combined["input_batches"], dim=0)
        combined["input_masks"]   = torch.cat(combined["input_masks"], dim=0)
        combined["label_batches"] = torch.cat(combined["label_batches"], dim=0)
        combined["label_masks"]   = torch.cat(combined["label_masks"], dim=0)
        return combined

# ---------- 训练循环 ------------
def train_one_epoch(model, dataloader, optimizer, device,
                    pitch_vocab_out, onset_vocab_out, duration_vocab_out, velocity_vocab_out,
                    epoch_idx=0, num_epochs=1):
    """
    在遍历 dataloader 的过程中使用 tqdm 显示进度条，并打印迭代间的 loss。
    """
    model.train()
    total_loss = 0.0
    total_steps = 0

    pitch_crit = torch.nn.CrossEntropyLoss(ignore_index=0)
    onset_crit = torch.nn.CrossEntropyLoss(ignore_index=0)
    dur_crit   = torch.nn.CrossEntropyLoss(ignore_index=0)
    vel_crit   = torch.nn.CrossEntropyLoss(ignore_index=0)

    # 构建 tqdm 进度条，显示当前 epoch 索引
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{num_epochs}", leave=True)

    for batch_data in progress_bar:
        input_batches = batch_data["input_batches"].to(device)   # [B, seq_len, 4]
        label_batches = batch_data["label_batches"].to(device)   # [B, seq_len, 4]
        # ...其他如 masks、处理自回归 mask 等

        # 前向传播
        outputs = model(input_batches, label_batches,
                        enc_mask=None, dec_mask=None, enc_dec_mask=None)

        # 计算 loss
        pitch_logits    = outputs["pitch"].permute(0,2,1)
        onset_logits    = outputs["onset"].permute(0,2,1)
        duration_logits = outputs["duration"].permute(0,2,1)
        velocity_logits = outputs["velocity"].permute(0,2,1)

        pitch_target    = label_batches[..., 0]  # [B, seq_len]
        onset_target    = label_batches[..., 1]
        duration_target = label_batches[..., 2]
        velocity_target = label_batches[..., 3]

        pitch_loss    = pitch_crit(pitch_logits, pitch_target)
        onset_loss    = onset_crit(onset_logits, onset_target)
        dur_loss      = dur_crit(duration_logits, duration_target)
        vel_loss      = vel_crit(velocity_logits, velocity_target)

        loss = pitch_loss + onset_loss + dur_loss + vel_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        total_steps += 1

        # 在 tqdm 进度条上显示当前 loss
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    return avg_loss


def main():
    # ---- 超参数设置 ----
    data_path = "midi_batches.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1   # 如果每个 dict 本身就是一个batch，可以用1; 否则适当调大
    num_epochs = 5
    lr = 1e-4

    # ---- 创建 DataLoader ----
    dataset = MidiDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # ---- 构建模型实例 ----
    # 需要根据实际的词表大小来设置 vocab_out
    # 这里仅示例
    pitch_vocab_in     = 128
    velocity_vocab_in  = 8
    pitch_vocab_out    = 128
    onset_vocab_out    = 5120
    duration_vocab_out = 1024
    velocity_vocab_out = 8

    model_dim = 512
    embed_dim = 64

    # 从 model.py 中导入的 MidiTransformer
    model = MidiTransformer(
        pitch_vocab_in, velocity_vocab_in,
        pitch_vocab_out, onset_vocab_out, duration_vocab_out, velocity_vocab_out,
        embed_dim, model_dim=model_dim,
        num_heads=4, ff_dim=2048,
        num_encoder_layers=2, num_decoder_layers=2,
        dropout=0.2, max_seq_len=1024
    )

    model = model.to(device)

    # ---- 优化器 ----
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---- 训练 Loop ----
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, device,
            pitch_vocab_out, onset_vocab_out, duration_vocab_out, velocity_vocab_out,
            epoch_idx=epoch, num_epochs=num_epochs  # 传入用于显示
        )
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

    # 可以在训练完后保存模型
    torch.save(model.state_dict(), "midi_transformer.pth")
    print("训练完成，模型已保存到 midi_transformer.pth")


if __name__ == "__main__":
    main()
