import os
import mido

def check_multiple_tempo_in_directory(directory: str):
    """
    在给定目录下（含子目录）查找所有 .mid/.midi 文件。
    对于每个文件，检查是否存在两个或以上 set_tempo 事件。
    如果存在，则在控制台打印相关文件路径和 tempo 事件数量。
    """
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            # 根据后缀判断是否为 MIDI 文件
            if file_name.lower().endswith((".mid", ".midi")):
                file_path = os.path.join(root, file_name)
                try:
                    mid = mido.MidiFile(file_path)
                    # 统计所有轨道里的 set_tempo 事件数量
                    tempo_count = sum(
                        1 for track in mid.tracks for msg in track if msg.type == 'set_tempo'
                    )

                    if tempo_count > 1:
                        print(f"[发现多Tempo] 文件: {file_path}, tempo事件数: {tempo_count}")
                except Exception as e:
                    print(f"[读取错误] 无法解析文件: {file_path}, 原因: {e}")


if __name__ == "__main__":
    # 使用示例：给函数传入要检查的文件夹路径
    # 例如想检查当前目录，可用: "."
    folder_to_check = "asap-dataset"
    check_multiple_tempo_in_directory(folder_to_check)
