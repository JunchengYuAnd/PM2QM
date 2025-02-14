import mido
import matplotlib.pyplot as plt
from typing import List, Tuple

# 定义 Token 类型：(pitch, onset_shift, duration_token, velocity_token)
Token = Tuple[int, int, int, int]


def get_tempo(mid: mido.MidiFile) -> int:
    """
    获取 MIDI 中第一个 set_tempo 事件的节奏（微秒/四分音符），如果没有则返回默认节奏 500000（即 120 BPM）。
    """
    return next(
        (msg.tempo for track in mid.tracks for msg in track if msg.type == 'set_tempo'),
        500000
    )


def tick_to_seconds(tick: int, ppq: int, tempo: int) -> float:
    """
    将 tick 转换为秒。
    :param tick: MIDI tick 数
    :param ppq: 每个四分音符的 tick 数
    :param tempo: 每个四分音符的微秒数
    :return: 对应的秒数
    """
    return (tick / ppq) * (tempo / 1e6)


def midi_to_tokens(midi_path: str, quantization: float = 0.01, velocity_bins: int = 8) -> List[Token]:
    """
    解析 MIDI 文件，并将 Note 事件转换为 Token 格式。

    每个 Token 格式为 (pitch, onset_shift, duration_token, velocity_token)：
    - onset_shift 和 duration 都按 quantization 进行量化（单位：秒）
    - velocity 归一化到指定的 velocity_bins 数级（默认为 32 级）

    :param midi_path: MIDI 文件路径
    :param quantization: 量化单位，默认 0.01 秒
    :param velocity_bins: 力度量化级数，默认 32
    :return: Token 列表
    """
    mid = mido.MidiFile(midi_path)
    ppq = mid.ticks_per_beat
    tempo = get_tempo(mid)

    events = []
    active_notes = {}  # 用于存储活跃的 Note-On 事件

    # 遍历所有轨道和消息，计算绝对时间并匹配 Note-On 与 Note-Off 事件
    for track in mid.tracks:
        track_time = 0.0
        for msg in track:
            track_time += tick_to_seconds(msg.time, ppq, tempo)

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (track_time, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    onset_time, note_velocity = active_notes.pop(msg.note)
                    duration = track_time - onset_time
                    events.append((msg.note, onset_time, duration, note_velocity))

    # 按照 Note-On 时间排序
    events.sort(key=lambda x: x[1])

    tokens = []
    for pitch, onset, duration, velocity in events:
        onset_shift = round(onset / quantization)
        duration_token = round(duration / quantization)
        velocity_token = round((velocity / 127) * (velocity_bins - 1))
        tokens.append((pitch, onset_shift, duration_token, velocity_token))

    return tokens


def split_into_batches(tokens: List[Token], batch_size: int = 128, overlap: int = 16) -> List[List[Token]]:
    """
    将 Token 序列划分为固定大小的批次，每个批次包含 batch_size 个 Token，
    并确保相邻批次之间有 overlap 个 Token 的重叠。
    批次内部会以第一个 Token 的 onset_shift 作为基准对所有 onset_shift 进行归一化，
    如果最后一批不足 batch_size，则使用 (0, 0, 0, 0) 填充。

    :param tokens: 原始 Token 序列
    :param batch_size: 每个批次包含的 Token 数量，默认为 128
    :param overlap: 批次之间重叠的 Token 数量，默认为 16
    :return: 划分后的批次列表
    """
    batches = []
    num_tokens = len(tokens)
    step_size = batch_size - overlap

    for i in range(0, num_tokens, step_size):
        batch = tokens[i:i + batch_size]
        if not batch:
            continue

        # 使用第一个 Token 的 onset_shift 作为基准
        batch_start = batch[0][1]
        adjusted_batch = [(p, onset - batch_start, d, v) for p, onset, d, v in batch]

        # 如果不足 batch_size，则用 (0,0,0,0) 填充
        if len(adjusted_batch) < batch_size:
            adjusted_batch.extend([(0, 0, 0, 0)] * (batch_size - len(adjusted_batch)))
        batches.append(adjusted_batch)

    return batches


def plot_batches(batches: List[List[Token]], quantization: float = 0.01):
    """
    可视化每个批次内的 MIDI Token 序列，
    使用水平线表示音符的持续时间，横轴为时间（秒），纵轴为音高。

    :param batches: 批次列表，每个批次为 Token 列表
    :param quantization: 量化单位（秒），用于将 Token 的数值转换为时间
    """
    num_batches = len(batches)
    fig, axes = plt.subplots(num_batches, 1, figsize=(10, 2 * num_batches), sharex=True)
    if num_batches == 1:
        axes = [axes]

    for ax, batch in zip(axes, batches):
        # 这里直接将量化后的 onset_shift 和 duration 用于可视化
        onset_times = [token[1] * quantization for token in batch]
        durations = [token[2] * quantization for token in batch]
        pitches = [token[0] for token in batch]

        for start, dur, pitch in zip(onset_times, durations, pitches):
            ax.hlines(y=pitch, xmin=start, xmax=start + dur, color='b', linewidth=2)
        ax.set_ylabel("Pitch")
    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    midi_path = "asap-dataset/Liszt/Annees_de_pelerinage_2/1_Gondoliera/LeungM08M.mid"
    tokens = midi_to_tokens(midi_path)
    batches = split_into_batches(tokens)
    print(f"Total batches: {len(batches)}")

    # 输出前 3 个批次的部分 Token 数据
    for i, batch in enumerate(batches[0:]):
        print(f"Batch {i + 1} (first 10 tokens): {batch[:10]}")
        print(f"Batch length: {len(batch)}")

    plot_batches(batches)