"""
这个文件用于数据预处理，实现--
1.遍历Train_set文件夹，自动识别所有脑电数据文件（如ST7011J0_EEGFpz Cz Part1 of 5.txt）
和标签文件（如ST7011J0_Hypnogram_Part1 of 5.txt）；
2.按 “被试 ID+Part_X” 匹配对应的脑电数据和标签（如 Part1 数据对应 Part1 标签）；
3.读取标签文件，筛选出有效标签（仅保留 R/1/2/3，删除?）；
4.验证结果，确保数据和标签数量对应。
5.按照要求进行帧分割并滤波
"""
import os
import numpy as np
import pandas as pd
from scipy import signal
# ====================== 第一步：配置路径 ======================

# 替换为本地的Train_set路径！！！
data_path = r"../data/fake_data"

# ====================== 第二步：遍历文件夹，分类数据文件和标签文件 ======================
# 存储所有文件名称
all_files = os.listdir(data_path)
# 分类：脑电数据文件（EEGFpz Cz）、标签文件（Hypnogram）
eeg_files = []  # 脑电数据文件列表
label_files = []  # 标签文件列表

for file_name in all_files:
    # 过滤掉非txt文件（防止文件夹中有其他格式文件）
    if not file_name.endswith(".txt"):
        continue
    # 区分脑电数据和标签文件
    if "EEGFpz Cz" in file_name:
        eeg_files.append(file_name)
    elif "Hypnogram" in file_name:
        label_files.append(file_name)

# 打印文件数量，验证分类成功
print(f"找到 {len(eeg_files)} 个脑电数据文件")
print(f"找到 {len(label_files)} 个标签文件")
print("=" * 60)


# ====================== 第三步：按被试ID+Part匹配数据和标签 ======================
# 定义匹配函数：从文件名中提取“被试ID+Part_X”（如ST7011J0_Part1）
def get_match_key(file_name):
    # 示例文件名：ST7011J0_EEGFpz Cz Part1 of 5.txt
    # 提取ST7011J0 + Part1
    id_part = file_name.split("_")[0]  # 提取被试ID（如ST7011J0）
    part_str = file_name.split("Part")[1].split(" of")[0]  # 提取Part后的数字（如1）
    match_key = f"{id_part}_Part{part_str}"
    return match_key


# 构建标签文件的匹配字典（key: 被试ID_PartX, value: 标签文件名）
label_key_dict = {}
for label_file in label_files:
    key = get_match_key(label_file)
    label_key_dict[key] = label_file

# 遍历脑电文件，匹配对应的标签文件
matched_data = []  # 存储匹配成功的（脑电文件, 标签文件）
unmatched_files = []  # 存储匹配失败的文件（备用排查）

for eeg_file in eeg_files:
    key = get_match_key(eeg_file)
    if key in label_key_dict:
        # 匹配成功
        label_file = label_key_dict[key]
        matched_data.append({
            "eeg_file": eeg_file,
            "label_file": label_file,
            "match_key": key
        })
    else:
        unmatched_files.append(eeg_file)

# 打印匹配结果
print(f"成功匹配 {len(matched_data)} 组数据-标签文件")
if unmatched_files:
    print(f"匹配失败的文件：{unmatched_files}")
else:
    print("所有脑电文件都匹配到了对应的标签文件！")
print("=" * 60)

# ====================== 第四步：读取标签文件，筛选有效标签 ======================
# 定义有效标签列表
valid_labels = ["R", "1", "2", "3"]
# 用于存储所有文件的处理结果
all_processed_data = []
if matched_data:
    # 遍历所有匹配成功的文件组，批量处理
    for idx, match_item in enumerate(matched_data, 1):
        eeg_file = match_item["eeg_file"]
        label_file = match_item["label_file"]
        match_key = match_item["match_key"]

        # 读取脑电数据
        eeg_file_path = os.path.join(data_path, eeg_file)
        eeg_data = np.loadtxt(eeg_file_path)

        # 读取标签文件（标签文件是文本格式，需按列分割）
        label_file_path = os.path.join(data_path, label_file)
        label_data = pd.read_csv(label_file_path, sep="\s+", header=None)
        label_data.columns = ["start_time", "end_time", "duration", "label"]

        # 筛选有效标签：删除?，仅保留R/1/2/3
        clean_label_data = label_data[label_data["label"].isin(valid_labels)]

        # 验证：有效标签的总时长（秒）是否和脑电数据时长匹配
        eeg_duration = len(eeg_data) / 100  # 采样频率100Hz
        label_total_duration = clean_label_data["duration"].sum()

        # 打印当前文件的处理结果
        print(f"【第{idx}组】匹配键：{match_key}")
        print(f"脑电文件：{eeg_file} | 数据点数量：{len(eeg_data)} | 时长：{eeg_duration:.1f}秒")
        print(f"标签文件：{label_file} | 原始标签行：{len(label_data)} | 有效标签行：{len(clean_label_data)}")
        print(
            f"有效标签总时长：{label_total_duration:.1f}秒 | 时长偏差：{abs(eeg_duration - label_total_duration):.1f}秒")
        print("-" * 40)

        # 存储当前文件的处理结果（含原始数据和清洗后标签，后续步骤可直接调用）
        all_processed_data.append({
            "match_key": match_key,
            "eeg_file": eeg_file,
            "eeg_data": eeg_data,
            "label_file": label_file,
            "raw_label": label_data,
            "clean_label": clean_label_data,
            "eeg_duration": eeg_duration,
            "label_total_duration": label_total_duration
        })

        # 打印批量处理总统计
    print(f"\n批量处理完成！共处理 {len(all_processed_data)} 组有效数据-标签文件")
else:
    print("无匹配成功的文件，无法进行标签清洗！")

print("\n第二步任务完成：批量读取+数据-标签匹配+标签清洗！")


# ====================== 第三步：脑电信号滤波 + 30秒帧分割 ======================
# ---------------------- 3.1 定义滤波函数 ----------------------
def filter_eeg_signal(eeg_data, fs=100):
    """
    对脑电信号进行滤波：0.5-30Hz带通滤波 + 50Hz陷波滤波
    :param eeg_data: 原始脑电数据（一维numpy数组）
    :param fs: 采样频率（已按照题目要求设置为100hz）
    :return: 滤波后的脑电数据
    """
    # 1. 50Hz陷波滤波（去除工频干扰）
    f0 = 50.0  # 要去除的工频频率
    Q = 30.0  # 品质因数，越大滤波越窄
    b, a = signal.iirnotch(f0, Q, fs)
    eeg_notch = signal.filtfilt(b, a, eeg_data)  # 零相位滤波，避免信号偏移

    # 2. 0.5-30Hz带通滤波（保留睡眠相关脑电波）
    low = 0.5  # 低频截止
    high = 30.0  # 高频截止
    # 调用4阶巴特沃斯带通滤波器
    b, a = signal.butter(4, [low, high], btype='bandpass', fs=fs)
    eeg_filtered = signal.filtfilt(b, a, eeg_notch)  # 零相位滤波

    return eeg_filtered


# ---------------------- 3.2 定义30秒帧分割函数 ----------------------
def split_eeg_into_frames(eeg_data, clean_label, fs=100, frame_duration=30):
    """
    将滤波后的脑电数据按30秒/帧分割，与清洗后的标签一一对应
    :param eeg_data: 滤波后的脑电数据
    :param clean_label: 清洗后的标签数据（DataFrame）
    :param fs: 采样频率（100Hz）
    :param frame_duration: 帧时长（30秒）
    :return: frames（脑电帧列表）, labels（对应标签列表）
    """
    frame_points = fs * frame_duration  # 每帧数据点：100*30=3000
    frames = []
    labels = []

    # 遍历清洗后的标签，按时间截取对应脑电帧
    for idx, row in clean_label.iterrows():
        start_sec = row["start_time"]
        end_sec = row["end_time"]

        # 计算对应数据点的索引（向下取整，避免越界）
        start_idx = int(np.floor(start_sec * fs))
        end_idx = int(np.floor(end_sec * fs))

        # 截取当前帧的脑电数据（确保是3000个点）
        if end_idx - start_idx == frame_points:
            frame = eeg_data[start_idx:end_idx]
            frames.append(frame)
            labels.append(row["label"])
        # 若长度不足/超出，跳过（保证每帧都是3000个点）
        else:
            print(f"⚠️ 帧{idx}长度异常：{end_idx - start_idx}个点（需3000），已跳过")

    return np.array(frames), np.array(labels)


# ---------------------- 3.3 批量处理：滤波 + 帧分割 ----------------------
# 存储最终可用于特征提取的数据（所有文件的帧+标签）
final_train_data = []

if all_processed_data:
    for idx, item in enumerate(all_processed_data, 1):
        match_key = item["match_key"]
        raw_eeg = item["eeg_data"]
        clean_label = item["clean_label"]

        print(f"【第{idx}组】处理：{match_key}")

        # 步骤1：滤波
        filtered_eeg = filter_eeg_signal(raw_eeg)
        print(f"✅ 滤波完成 | 原始数据长度：{len(raw_eeg)} | 滤波后长度：{len(filtered_eeg)}")

        # 步骤2：30秒帧分割
        frames, labels = split_eeg_into_frames(filtered_eeg, clean_label)
        print(f"✅ 帧分割完成 | 有效帧数量：{len(frames)} | 对应标签数量：{len(labels)}")
        print("-" * 40)

        # 存储最终结果（按照要求整理）
        final_train_data.append({
            "match_key": match_key,
            "filtered_eeg": filtered_eeg,  # 全量滤波后数据
            "eeg_frames": frames,  # 30秒帧（shape: [n_frames, 3000]）
            "frame_labels": labels,  # 帧对应标签（shape: [n_frames]）
            "frame_count": len(frames)  # 帧数量
        })

    # 打印总统计
    total_frames = sum([item["frame_count"] for item in final_train_data])
    print(f"\n📊 滤波+帧分割全量处理完成！")
    print(f"总处理文件组数：{len(final_train_data)}")
    print(f"总有效30秒帧数量：{total_frames}")
    print(f"每帧数据点：{30 * 100}个（符合要求）")
else:
    print("❌ 无预处理数据，无法进行滤波和帧分割！")

print("\n第三步任务完成：脑电滤波 + 30秒帧分割！")