"""
这个文件用于数据预处理，实现--
1.遍历Train_set文件夹，自动识别所有脑电数据文件（如ST7011J0_EEGFpz Cz Part1 of 5.txt）
和标签文件（如ST7011J0_Hypnogram_Part1 of 5.txt）；
2.按 “被试 ID+Part_X” 匹配对应的脑电数据和标签（如 Part1 数据对应 Part1 标签）；
3.读取标签文件，筛选出有效标签（仅保留 R/1/2/3，删除?/W/其他）；
4.验证结果，确保数据和标签数量对应。
5.按照要求进行帧分割并滤波
"""
import os
import re
import numpy as np
import pandas as pd
from scipy import signal

# ====================== 第一步：配置路径 ======================
# 替换为本地的Train_set路径！！！
data_path = r"..\data\train_data"

# ====================== 第二步：遍历文件夹，分类数据文件和标签文件 ======================
# 存储所有文件名称
all_files = os.listdir(data_path)
# 分类：脑电数据文件（EEGFpz_Cz）、标签文件（Hypnogram）
eeg_files = []  # 脑电数据文件列表
label_files = []  # 标签文件列表

for file_name in all_files:
    # 过滤条件1：仅保留txt文件
    if not file_name.endswith(".txt"):
        continue
    # 过滤条件2：排除滤波后的冗余文件（含filtered关键词）
    if "filtered" in file_name.lower():
        continue
    # 区分脑电数据和标签文件
    if "EEGFpz_Cz" in file_name:
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
    """
    鲁棒性匹配：兼容不同文件名格式（Part_1 / Part1）
    """
    try:
        # 提取被试ID（下划线分割第一个部分）
        id_part = file_name.split("_")[0]
        # 兼容两种格式：Part1 of 5 或 Part_1 of 5
        part_match = re.search(r'Part[_ ]*(\d+)', file_name)
        if part_match:
            part_str = part_match.group(1)
            match_key = f"{id_part}_Part{part_str}"
            return match_key
        else:
            print(f"⚠️ 未找到Part编号：{file_name}")
            return None
    except Exception as e:
        print(f"⚠️ 文件名解析失败：{file_name} | 错误：{e}")
        return None


# 构建标签文件的匹配字典（key: 被试ID_PartX, value: 标签文件名）
label_key_dict = {}
for label_file in label_files:
    key = get_match_key(label_file)
    if key:  # 仅存储解析成功的标签文件
        label_key_dict[key] = label_file

# 遍历脑电文件，匹配对应的标签文件
matched_data = []  # 存储匹配成功的（脑电文件, 标签文件）
unmatched_files = []  # 存储匹配失败的文件（备用排查）

for eeg_file in eeg_files:
    key = get_match_key(eeg_file)
    if key and key in label_key_dict:
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
    print(f"匹配失败的文件：{unmatched_files[:5]}")  # 仅打印前5个，避免输出过长
    print(f"匹配失败文件总数：{len(unmatched_files)}")
else:
    print("所有脑电文件都匹配到了对应的标签文件！")
print("=" * 60)

# ====================== 第四步：读取标签文件，筛选有效标签（逐行精准解析） ======================
# 定义标签映射关系：适配所有可能的标签格式
LABEL_MAP = {
    "Sleep stage R": "R",
    "Sleep stage 1": "1",
    "Sleep stage 2": "2",
    "Sleep stage 3": "3",
    "R": "R",
    "1": "1",
    "2": "2",
    "3": "3",
    "REM": "R",
    "N1": "1",
    "N2": "2",
    "N3": "3"
}
# 有效原始标签（用于筛选）
VALID_RAW_LABELS = list(LABEL_MAP.keys())

# 用于存储所有文件的处理结果
all_processed_data = []


def read_label_file_precise(file_path):
    """
    逐行精准读取标签文件：
    1. 自动跳过表头/空行/注释行
    2. 解析任意分隔符（空格/制表符/逗号）
    3. 提取时间和标签信息
    """
    label_rows = []
    # 先读取所有行，查看格式（调试用）
    all_lines = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        all_lines = [line.strip() for line in f if line.strip()]

    # 打印前3行，帮助调试标签格式
    print(f"\n🔍 标签文件格式调试 - {os.path.basename(file_path)}:")
    for i, line in enumerate(all_lines[:3]):
        print(f"   第{i + 1}行：{line}")

    # 逐行解析
    for line_num, line in enumerate(all_lines, 1):
        try:
            # 跳过表头行（包含onset/start/time等关键词）
            if any(keyword in line.lower() for keyword in
                   ['onset', 'start', 'end', 'duration', 'label', 'description']):
                continue

            # 用正则分割任意空白符/逗号
            parts = re.split(r'[\s,]+', line)
            # 过滤空字符串
            parts = [p for p in parts if p]

            # 解析时间和标签（兼容不同列数）
            if len(parts) >= 4:
                # 标准格式：start end duration label
                start_time = float(parts[0])
                end_time = float(parts[1])
                duration = float(parts[2])
                label_str = ' '.join(parts[3:])  # 标签可能包含空格（如Sleep stage R）
            elif len(parts) == 3:
                # 无duration格式：start end label
                start_time = float(parts[0])
                end_time = float(parts[1])
                duration = end_time - start_time
                label_str = parts[2]
            else:
                print(f"⚠️ 第{line_num}行格式异常（列数不足）：{line}")
                continue

            # 清理标签字符串
            label_str = label_str.strip().replace('"', '').replace("'", "")
            label_rows.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "label": label_str
            })
        except Exception as e:
            print(f"⚠️ 第{line_num}行解析失败：{line} | 错误：{e}")
            continue

    # 转为DataFrame
    label_df = pd.DataFrame(label_rows)
    return label_df


if matched_data:
    # 先处理前1个文件做调试，确认标签格式
    test_item = matched_data[0]
    test_label_path = os.path.join(data_path, test_item["label_file"])
    test_label_df = read_label_file_precise(test_label_path)

    # 遍历所有匹配成功的文件组，批量处理
    for idx, match_item in enumerate(matched_data, 1):
        eeg_file = match_item["eeg_file"]
        label_file = match_item["label_file"]
        match_key = match_item["match_key"]

        try:
            # 读取脑电数据（跳过非数值行，确保一维数组）
            eeg_file_path = os.path.join(data_path, eeg_file)
            eeg_data = np.loadtxt(eeg_file_path, ndmin=1)  # 确保是一维数组

            # 读取标签文件（精准解析）
            label_file_path = os.path.join(data_path, label_file)
            label_data = read_label_file_precise(label_file_path)

            if label_data.empty:
                print(f"⚠️ 【第{idx}组】{match_key} 标签文件为空，跳过")
                continue

            # 打印原始标签值，帮助匹配
            unique_labels = label_data["label"].unique()
            print(f"📌 【第{idx}组】{match_key} 原始标签值：{unique_labels}")


            # 核心修复：筛选有效标签 + 映射
            # 1. 先筛选包含有效关键词的标签
            def is_valid_label(lbl):
                lbl_lower = lbl.lower()
                return any(keyword in lbl_lower for keyword in ['r', '1', '2', '3', 'rem', 'n1', 'n2', 'n3'])


            # 2. 筛选有效行
            clean_label_data = label_data[label_data["label"].apply(is_valid_label)].copy()


            # 3. 映射标签（兼容多种格式）
            def map_label(lbl):
                # 先尝试精确匹配
                if lbl in LABEL_MAP:
                    return LABEL_MAP[lbl]
                # 模糊匹配
                lbl_lower = lbl.lower()
                if 'r' in lbl_lower or 'rem' in lbl_lower:
                    return 'R'
                elif '1' in lbl_lower or 'n1' in lbl_lower:
                    return '1'
                elif '2' in lbl_lower or 'n2' in lbl_lower:
                    return '2'
                elif '3' in lbl_lower or 'n3' in lbl_lower:
                    return '3'
                else:
                    return None


            clean_label_data["label"] = clean_label_data["label"].apply(map_label)
            # 移除映射失败的标签
            clean_label_data = clean_label_data.dropna(subset=["label"])

            # 验证：有效标签的总时长（秒）是否和脑电数据时长匹配
            eeg_duration = len(eeg_data) / 100  # 采样频率100Hz
            label_total_duration = clean_label_data["duration"].sum()

            # 打印当前文件的处理结果
            print(f"\n✅ 【第{idx}组】匹配键：{match_key}")
            print(f"   脑电文件：{eeg_file} | 数据点数量：{len(eeg_data)} | 时长：{eeg_duration:.1f}秒")
            print(f"   标签文件：{label_file} | 原始标签行：{len(label_data)} | 有效标签行：{len(clean_label_data)}")
            print(
                f"   有效标签总时长：{label_total_duration:.1f}秒 | 时长偏差：{abs(eeg_duration - label_total_duration):.1f}秒")
            if len(clean_label_data) > 0:
                print(f"   有效标签分布：{clean_label_data['label'].value_counts().to_dict()}")
            print("-" * 60)

            # 存储当前文件的处理结果
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

        except Exception as e:
            print(f"❌ 【第{idx}组】处理失败：{match_key} | 错误：{e}")
            import traceback

            traceback.print_exc()
            continue

    # 打印批量处理总统计
    print(f"\n📊 批量处理完成！共处理 {len(all_processed_data)} 组有效数据-标签文件")
    total_valid_label_lines = sum([len(item["clean_label"]) for item in all_processed_data])
    total_valid_duration = sum([item["label_total_duration"] for item in all_processed_data])
    print(f"总有效标签行数：{total_valid_label_lines} | 总有效标签时长：{total_valid_duration:.1f}秒")

    # 统计所有有效标签分布
    all_labels = []
    for item in all_processed_data:
        all_labels.extend(item["clean_label"]["label"].tolist())
    if all_labels:
        label_counts = pd.Series(all_labels).value_counts()
        print(f"总标签分布：{label_counts.to_dict()}")
else:
    print("无匹配成功的文件，无法进行标签清洗！")

print("\n第四步任务完成：批量读取+数据-标签匹配+标签清洗！")


# ====================== 第五步：脑电信号滤波 + 30秒帧分割 ======================
# ---------------------- 5.1 定义滤波函数 ----------------------
def filter_eeg_signal(eeg_data, fs=100):
    """
    对脑电信号进行滤波：0.5-30Hz带通滤波 + 50Hz陷波滤波
    """
    # 空数据直接返回
    if len(eeg_data) == 0:
        return eeg_data

    # 1. 50Hz陷波滤波（去除工频干扰）
    f0 = 50.0  # 要去除的工频频率
    Q = 27.5 # 品质因数，越大滤波越窄
    b, a = signal.iirnotch(f0, Q, fs)
    eeg_notch = signal.filtfilt(b, a, eeg_data)  # 零相位滤波，避免信号偏移

    # 2. 0.5-30Hz带通滤波（保留睡眠相关脑电波）
    low = 0.5  # 低频截止
    high = 30.0  # 高频截止
    # 调用5阶巴特沃斯带通滤波器
    b, a = signal.butter(5, [low, high], btype='bandpass', fs=fs)
    eeg_filtered = signal.filtfilt(b, a, eeg_notch)  # 零相位滤波

    return eeg_filtered


# ---------------------- 5.2 定义30秒帧分割函数 ----------------------
def split_eeg_into_frames(eeg_data, clean_label, fs=100, frame_duration=30):
    """
    将滤波后的脑电数据按30秒/帧分割，与清洗后的标签一一对应
    """
    frame_points = fs * frame_duration  # 每帧数据点：100*30=3000
    frames = []
    labels = []

    # 空数据/空标签直接返回
    if len(eeg_data) == 0 or clean_label.empty:
        return np.array(frames), np.array(labels)

    # 遍历清洗后的标签，按时间截取对应脑电帧
    for idx, row in clean_label.iterrows():
        try:
            start_sec = float(row["start_time"])
            end_sec = float(row["end_time"])
            current_label = row["label"]

            # 计算对应数据点的索引（向下取整，避免越界）
            start_idx = int(np.floor(start_sec * fs))
            end_idx = int(np.floor(end_sec * fs))

            # 边界检查：避免索引越界
            if start_idx < 0 or end_idx > len(eeg_data):
                continue

            # 处理非30秒的标签段：按30秒切分
            segment_duration = end_sec - start_sec
            if segment_duration >= frame_duration:
                # 按30秒步长切分长标签段
                for sub_start in np.arange(start_sec, end_sec, frame_duration):
                    sub_end = sub_start + frame_duration
                    if sub_end > end_sec:
                        break
                    # 计算子段索引
                    sub_start_idx = int(np.floor(sub_start * fs))
                    sub_end_idx = int(np.floor(sub_end * fs))
                    if sub_end_idx - sub_start_idx == frame_points:
                        frame = eeg_data[sub_start_idx:sub_end_idx]
                        frames.append(frame)
                        labels.append(current_label)
            # 刚好30秒的段，直接截取
            elif segment_duration == frame_duration:
                if end_idx - start_idx == frame_points:
                    frame = eeg_data[start_idx:end_idx]
                    frames.append(frame)
                    labels.append(current_label)

        except Exception as e:
            continue

    return np.array(frames), np.array(labels)


# ---------------------- 5.3 批量处理：滤波 + 帧分割 ----------------------
# 存储最终可用于特征提取的数据（所有文件的帧+标签）
final_train_data = []

if all_processed_data:
    for idx, item in enumerate(all_processed_data, 1):
        match_key = item["match_key"]
        raw_eeg = item["eeg_data"]
        clean_label = item["clean_label"]

        try:
            # 每5组打印一次处理进度
            if idx % 20 == 0:
                print(f"\n【第{idx}组】处理滤波+帧分割：{match_key}")

            # 步骤1：滤波
            filtered_eeg = filter_eeg_signal(raw_eeg)
            if idx % 20 == 0:
                print(f"✅ 滤波完成 | 原始数据长度：{len(raw_eeg)} | 滤波后长度：{len(filtered_eeg)}")

            # 步骤2：30秒帧分割
            frames, labels = split_eeg_into_frames(filtered_eeg, clean_label)
            if idx % 20 == 0:
                print(f"✅ 帧分割完成 | 有效帧数量：{len(frames)} | 对应标签数量：{len(labels)}")
                if len(labels) > 0:
                    label_counts = pd.Series(labels).value_counts()
                    print(f"   标签分布：{label_counts.to_dict()}")
                print("-" * 60)

            # 存储最终结果
            final_train_data.append({
                "match_key": match_key,
                "filtered_eeg": filtered_eeg,
                "eeg_frames": frames,
                "frame_labels": labels,
                "frame_count": len(frames),
                "label_dist": pd.Series(labels).value_counts().to_dict() if len(labels) > 0 else {}
            })

        except Exception as e:
            print(f"❌ 【第{idx}组】滤波/帧分割失败：{match_key} | 错误：{e}")
            continue

    # 打印滤波+帧分割总统计
    total_frames = sum([item["frame_count"] for item in final_train_data])
    print(f"\n📊 滤波+帧分割全量处理完成！")
    print(f"总处理文件组数：{len(final_train_data)}")
    print(f"总有效30秒帧数量：{total_frames}")

    # 统计所有标签
    all_frame_labels = []
    for item in final_train_data:
        all_frame_labels.extend(item["frame_labels"].tolist())
    if all_frame_labels:
        total_label_counts = pd.Series(all_frame_labels).value_counts()
        print(f"总帧标签分布：{total_label_counts.to_dict()}")
    print(f"每帧数据点：{30 * 100}个（符合竞赛要求）")
else:
    print("❌ 无预处理数据，无法进行滤波和帧分割！")

# 保存预处理后的最终数据
save_path = r"..\data\processed_data"
if not os.path.exists(save_path):
    os.makedirs(save_path)

if final_train_data and total_frames > 0:
    # 合并所有帧和标签
    all_frames = np.concatenate([item["eeg_frames"] for item in final_train_data if len(item["eeg_frames"]) > 0])
    all_labels = np.concatenate([item["frame_labels"] for item in final_train_data if len(item["frame_labels"]) > 0])

    # 保存为npy文件
    np.save(os.path.join(save_path, "all_eeg_frames.npy"), all_frames)
    np.save(os.path.join(save_path, "all_frame_labels.npy"), all_labels)
    print(f"\n✅ 预处理后的数据已保存至：{save_path}")
    print(f"   - 脑电帧：all_eeg_frames.npy (shape: {all_frames.shape})")
    print(f"   - 标签：all_frame_labels.npy (shape: {all_labels.shape})")

print("\n第五步任务完成：脑电滤波 + 30秒帧分割！")