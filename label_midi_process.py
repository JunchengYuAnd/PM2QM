import mido
from typing import List, Tuple
import matplotlib.pyplot as plt

# 自定义 Token 格式：(pitch, onset_shift, duration_token, velocity_token)
Token = Tuple[int, int, int, int]

def get_single_tempo(mid: mido.MidiFile) -> int:
    """
    在给定的 MidiFile 对象中找到第一个 tempo，如果文件里有多个 tempo，
    仅保留第一个，覆盖后续所有的 set_tempo 为同一个值。
    若未找到任何 tempo，则使用 500000 (相当于120 BPM)。

    :param mid: Mido 读入的 MidiFile 对象
    :return: 确定的（单一）tempo 整数值
    """
    found_tempo = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                if found_tempo is None:
                    found_tempo = msg.tempo  # 记录第一个tempo
                else:
                    # 覆盖后续tempo事件
                    msg.tempo = found_tempo

    # 如果整首曲子都没有找到 set_tempo，则设置为 500000（120 BPM）
    if found_tempo is None:
        found_tempo = 500000

    return found_tempo


def tick_to_seconds(tick: int, ppq: int, tempo: int) -> float:
    """
    将 tick 转换为秒。
    :param tick: MIDI tick 数
    :param ppq: 每个四分音符的 tick 数 (ticks_per_beat)
    :param tempo: 每个四分音符对应的微秒数
    :return: 对应的秒数
    """
    # tempo(微秒/四分音符)， 1e6 微秒 = 1 秒
    return (tick / ppq) * (tempo / 1e6)


def midi_to_tokens(midi_path: str, velocity_bins: int = 8) -> List[Token]:
    """
    读取 MIDI 文件，并转换为基于 1/24 拍 量化的 Token 序列。

    每个 Token 格式为 (音高, 量化起始时间, 量化持续时间, 量化力度)，其中：
      - 量化单位 T_grid = tempo / (1e6 * 24) （单位：秒），tempo 在此处统一为单一值
      - 力度归一化为 0 ~ (velocity_bins - 1) 的整数

    :param midi_path: MIDI 文件路径
    :param velocity_bins: 力度量化的级数，默认 32
    :return: Token 序列列表
    """
    # 读取文件
    mid = mido.MidiFile(midi_path)
    ppq = mid.ticks_per_beat

    # 先统一 Tempo，让全曲只有一个速度
    tempo = get_single_tempo(mid)

    # 计算 1/24 拍 对应的秒数
    T_grid = tempo / (1e6 * 24)

    events: List[Token] = []
    active_notes = {}  # 用于存储 note_on 事件：键为音高，值为 (起始时间, 力度)

    # 遍历所有轨道，累加每个消息的 tick 转换后的绝对时间
    for track in mid.tracks:
        track_time = 0.0  # 以秒为单位的轨道内绝对时间
        for msg in track:
            # 根据 msg.time (相对tick) 更新轨道内绝对时间
            track_time += tick_to_seconds(msg.time, ppq, tempo)

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (track_time, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    onset_time, note_velocity = active_notes.pop(msg.note)
                    duration = track_time - onset_time
                    # 量化起始时间和持续时间
                    onset_shift = round(onset_time / T_grid)
                    duration_token = round(duration / T_grid)
                    # 力度归一化到 0 ~ (velocity_bins - 1)
                    velocity_token = round((note_velocity / 127) * (velocity_bins - 1))
                    events.append((msg.note, onset_shift, duration_token, velocity_token))

    # 根据量化起始时间排序
    events.sort(key=lambda x: x[1])
    return events




def split_into_batches(tokens: List[Token], batch_size: int = 128, overlap: int = 16) -> List[List[Token]]:
    """
    将 Token 序列分割为固定大小的批次，每个 batch 包含 batch_size 个 Token，
    并保证相邻批次之间有 overlap 个 Token 的重叠。
    同时对每个 batch 内的 Token 进行时间归一化（以该 batch 第一个 Token 为基准）。
    如果不足 batch_size，则用 (0, 0, 0, 0) 填充。

    :param tokens: Token 序列列表
    :param batch_size: 每个批次的 Token 数量（默认 128）
    :param overlap: 批次之间重叠的 Token 数量（默认 16）
    :return: 分割后的批次列表
    """
    batches = []
    num_tokens = len(tokens)
    step_size = batch_size - overlap  # 每次滑动的 Token 数量

    for i in range(0, num_tokens, step_size):
        batch = tokens[i:i + batch_size]
        if not batch:
            continue
        # 归一化：以该 batch 第一个 Token 的量化起始时间为基准
        batch_start = batch[0][1]
        adjusted_batch = [(p, onset - batch_start, d, v) for p, onset, d, v in batch]
        # 如果不足 batch_size，则进行填充
        if len(adjusted_batch) < batch_size:
            adjusted_batch.extend([(0, 0, 0, 0)] * (batch_size - len(adjusted_batch)))
        batches.append(adjusted_batch)
    return batches


def plot_batches(batches: List[List[Token]], quantization: float = 0.01):
    """
    可视化 MIDI Token 序列，每个批次用一个子图显示，音符以水平线表示。

    :param batches: Token 批次列表
    :param quantization: 用于将量化时间转换为秒的缩放因子（默认 0.01）
    """
    num_batches = len(batches)
    fig, axes = plt.subplots(num_batches, 1, figsize=(10, 2 * num_batches), sharex=True)
    if num_batches == 1:
        axes = [axes]

    for ax, batch in zip(axes, batches):
        # 提取各字段
        pitches = [token[0] for token in batch]
        # 注意：这里假设 token[1] 表示相对时间差（已归一化），用 np.cumsum 模拟时间流
        onset_times = [token[1] * quantization for token in batch]
        durations = [token[2] * quantization for token in batch]

        for start, dur, pitch in zip(onset_times, durations, pitches):
            # 为了更好显示，将持续时间放大一定比例（此处乘以 15）
            ax.hlines(y=pitch, xmin=start, xmax=start + dur, color='b', linewidth=2)
        ax.set_ylabel("Pitch")
    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例：处理标签数据的 MIDI 文件，并输出 Token 批次信息和可视化结果
    midi_path = "asap-dataset/Rachmaninoff/Preludes_op_32/10/midi_score.mid"
    tokens = midi_to_tokens(midi_path)
    batches = split_into_batches(tokens)
    for i, batch in enumerate(batches):
        print(f"Batch {i + 1} (first 10 tokens): {batch[:10]}")
        # 此处打印最后一个 Token 内所有字段的最小值（仅作为调试信息）
        print("Min value in last token:", min(batch[-1]))
        print("Batch length:", len(batch))
    plot_batches(batches)