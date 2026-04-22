"""
该文件用于测试集预测和睡眠质量分析，完成以下功能：
1. 读取训练好的模型和标签编码器
2. 读取Test_set中的脑电数据文件
3. 【重要】按受试者ID分组，将同一受试者的5个Part合并为整晚数据
4. 对测试数据进行滤波处理
5. 提取特征并进行睡眠分期预测
6. 计算各睡眠期时长，分析睡眠质量（基于整晚数据）
7. 生成可视化结果（睡眠结构图、质量评估图）
8. 按竞赛要求保存预测结果（每个Part单独保存）
"""

import os
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import joblib
from collections import defaultdict

warnings.filterwarnings('ignore')

# 设置中文字体，防止图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 第一步：配置路径 ======================

# 路径配置
test_data_path = r"..\data\test_data"
processed_data_path = r"..\data\processed_data"
results_path = r"..\results"
model_path = os.path.join(processed_data_path, "model")

# 创建results文件夹
if not os.path.exists(results_path):
    os.makedirs(results_path)
    print(f" 创建结果保存文件夹：{results_path}")

# ====================== 第二步：加载训练好的模型 ======================

try:
    print(" 加载训练好的模型...")
    rf_model = joblib.load(os.path.join(model_path, "sleep_stage_rf_model_optimized.pkl"))
    label_encoder = joblib.load(os.path.join(model_path, "label_encoder_optimized.pkl"))
    print(" 模型和标签编码器加载成功！")
    print(f"   - 标签编码映射：{dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    print("=" * 60)
except Exception as e:
    print(f"模型加载失败：{e}")
    print("   请确保已完成模型训练，并保存到指定路径！")
    exit()


# ====================== 第三步：复用预处理和特征提取函数 ======================

def filter_eeg_signal(eeg_data, fs=100):
    """
    对脑电信号进行滤波：0.5-30Hz带通滤波 + 50Hz陷波滤波
    """
    if len(eeg_data) == 0:
        return eeg_data

    # 50Hz陷波滤波
    f0 = 50.0
    Q = 27.5
    b, a = signal.iirnotch(f0, Q, fs)
    eeg_notch = signal.filtfilt(b, a, eeg_data)

    # 0.5-30Hz带通滤波
    low = 0.5
    high = 30.0
    b, a = signal.butter(5, [low, high], btype='bandpass', fs=fs)
    eeg_filtered = signal.filtfilt(b, a, eeg_notch)

    return eeg_filtered


def extract_eeg_features(frame, fs=100):
    """
    提取单帧脑电数据的特征（与训练时保持一致）
    """
    # 时域特征
    mean_val = np.mean(frame)
    std_val = np.std(frame)
    max_val = np.max(frame)
    min_val = np.min(frame)
    ptp_val = np.ptp(frame)
    rms_val = np.sqrt(np.mean(np.square(frame)))
    skewness_val = stats.skew(frame)
    kurtosis_val = stats.kurtosis(frame)
    zero_cross = np.sum(np.diff(np.sign(frame)) != 0)

    # 新增时域特征
    abs_frame = np.abs(frame)
    avg_abs = np.mean(abs_frame)
    peak_factor = max_val / rms_val if rms_val != 0 else 0
    waveform_factor = rms_val / avg_abs if avg_abs != 0 else 0
    impulse_factor = max_val / avg_abs if avg_abs != 0 else 0
    crest_factor = max_val / np.sqrt(np.mean(np.square(frame))) if rms_val != 0 else 0
    variance_val = np.var(frame)
    median_val = np.median(frame)
    q25_val = np.percentile(frame, 25)
    q75_val = np.percentile(frame, 75)
    iqr_val = q75_val - q25_val

    # 频域特征
    n = len(frame)
    freq = np.fft.fftfreq(n, 1 / fs)
    fft_vals = np.fft.fft(frame)
    power = np.abs(fft_vals) ** 2
    pos_mask = freq > 0
    freq_pos = freq[pos_mask]
    power_pos = power[pos_mask]

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30)
    }

    band_energy = {}
    band_mean = {}
    band_std = {}
    total_energy = np.sum(power_pos)
    if total_energy == 0:
        total_energy = 1e-6

    for band_name, (low, high) in bands.items():
        band_mask = (freq_pos >= low) & (freq_pos <= high)
        band_power = power_pos[band_mask]
        band_energy[band_name] = np.sum(band_power) / total_energy
        band_mean[band_name] = np.mean(band_power) if len(band_power) > 0 else 0
        band_std[band_name] = np.std(band_power) if len(band_power) > 0 else 0

    normalized_power = power_pos / total_energy
    spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
    total_power = np.sum(power_pos)

    features = [
        mean_val, std_val, max_val, min_val, ptp_val, rms_val,
        skewness_val, kurtosis_val, zero_cross,
        avg_abs, peak_factor, waveform_factor, impulse_factor, crest_factor,
        variance_val, median_val, q25_val, q75_val, iqr_val,
        band_energy["delta"], band_energy["theta"],
        band_energy["alpha"], band_energy["beta"],
        band_mean["delta"], band_mean["theta"], band_mean["alpha"], band_mean["beta"],
        band_std["delta"], band_std["theta"], band_std["alpha"], band_std["beta"],
        spectral_entropy, total_power
    ]

    return np.array(features)


from scipy import stats


# ====================== 第四步：读取并按受试者分组测试集数据 ======================

def get_test_files_grouped():
    """
    遍历test_data文件夹，获取所有脑电数据文件，并按受试者ID分组
    每个受试者有5个Part文件
    """
    if not os.path.exists(test_data_path):
        print(f" 测试数据路径不存在：{test_data_path}")
        return {}

    all_files = os.listdir(test_data_path)

    # 按受试者ID分组
    subject_files = defaultdict(list)

    for file_name in all_files:
        if not file_name.endswith(".txt"):
            continue
        if "filtered" in file_name.lower():
            continue
        if "EEGFpz_Cz" in file_name:
            # 提取受试者ID
            match = re.match(r'(.+?)_EEGFpz_Cz', file_name)
            if match:
                subject_id = match.group(1)
                subject_files[subject_id].append(file_name)

    # 对每个受试者的Part文件进行排序
    for subject_id in subject_files:
        subject_files[subject_id].sort()

    print(f" 找到 {len(subject_files)} 个受试者")
    for subject_id, files in subject_files.items():
        print(f"   - {subject_id}: {len(files)} 个Part文件")

    return subject_files


def predict_part_file(eeg_file):
    """
    对单个Part文件进行预测，返回预测结果和帧信息
    """
    file_path = os.path.join(test_data_path, eeg_file)
    eeg_data = np.loadtxt(file_path, ndmin=1)

    # 滤波
    filtered_eeg = filter_eeg_signal(eeg_data)

    # 按30秒分割帧
    fs = 100
    frame_duration = 30
    frame_points = fs * frame_duration

    total_frames = len(filtered_eeg) // frame_points
    frames = []

    for i in range(total_frames):
        start_idx = i * frame_points
        end_idx = start_idx + frame_points
        frame = filtered_eeg[start_idx:end_idx]
        if len(frame) == frame_points:
            frames.append(frame)

    if len(frames) == 0:
        print(f" {eeg_file} 没有有效的帧！")
        return None, None, 0

    # 提取特征
    feature_list = []
    for frame in frames:
        features = extract_eeg_features(frame)
        feature_list.append(features)

    features_matrix = np.array(feature_list)

    # 预测
    predictions = rf_model.predict(features_matrix)
    predicted_labels = label_encoder.inverse_transform(predictions)

    # 构建结果DataFrame
    results = []
    current_time = 0.0

    for label in predicted_labels:
        start_time = current_time
        end_time = current_time + frame_duration
        duration = frame_duration
        results.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "label": label
        })
        current_time = end_time

    return pd.DataFrame(results), eeg_file, len(frames)


# ====================== 第五步：睡眠质量分析（基于整晚数据） ======================

def analyze_sleep_quality(all_results_df_list, subject_id):
    """
    基于合并后的整晚睡眠数据（5个Part合并）分析睡眠质量
    all_results_df_list: 同一受试者所有Part的预测结果列表
    """
    if not all_results_df_list:
        return None

    # 合并所有Part的结果
    combined_results = pd.concat(all_results_df_list, ignore_index=True)

    # 重新计算时间轴（连续时间）
    for idx, row in combined_results.iterrows():
        combined_results.at[idx, 'start_time'] = idx * 30.0
        combined_results.at[idx, 'end_time'] = (idx + 1) * 30.0

    # 统计各睡眠期的帧数和时长
    label_stats = combined_results.groupby("label").agg({
        "duration": ["count", "sum"]
    }).reset_index()
    label_stats.columns = ["睡眠期", "帧数", "总时长(秒)"]

    # 计算总睡眠时长（排除清醒期W）
    sleep_stages = ["1", "2", "3", "R"]
    total_sleep_time = label_stats[label_stats["睡眠期"].isin(sleep_stages)]["总时长(秒)"].sum()

    # 计算各睡眠期占比
    label_stats["时长占比(%)"] = (label_stats["总时长(秒)"] / label_stats["总时长(秒)"].sum() * 100).round(2)

    # 计算睡眠质量指标
    # 1. 深睡眠比例（N3期）
    n3_duration = label_stats[label_stats["睡眠期"] == "3"]["总时长(秒)"].sum()
    deep_sleep_ratio = (n3_duration / total_sleep_time * 100) if total_sleep_time > 0 else 0

    # 2. 浅睡眠比例（N1+N2期）
    light_sleep_duration = label_stats[label_stats["睡眠期"].isin(["1", "2"])]["总时长(秒)"].sum()
    light_sleep_ratio = (light_sleep_duration / total_sleep_time * 100) if total_sleep_time > 0 else 0

    # 3. REM睡眠比例
    rem_duration = label_stats[label_stats["睡眠期"] == "R"]["总时长(秒)"].sum()
    rem_ratio = (rem_duration / total_sleep_time * 100) if total_sleep_time > 0 else 0

    # 4. 睡眠效率（实际睡眠时长/总记录时长）
    total_record_time = label_stats["总时长(秒)"].sum()
    sleep_efficiency = (total_sleep_time / total_record_time * 100) if total_record_time > 0 else 0

    # 5. 睡眠连续性（睡眠期转换次数，越少表示睡眠越连续）
    stage_transitions = 0
    labels = combined_results["label"].values
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            stage_transitions += 1

    # 6. 睡眠质量评分（基于各指标的综合评分）
    # 深睡眠（15-25%为优）、REM睡眠（20-25%为优）、睡眠效率（>85%为优）
    quality_score = 0
    if 15 <= deep_sleep_ratio <= 25:
        quality_score += 30
    elif 10 <= deep_sleep_ratio < 15:
        quality_score += 20
    elif 5 <= deep_sleep_ratio < 10:
        quality_score += 10

    if 20 <= rem_ratio <= 25:
        quality_score += 30
    elif 15 <= rem_ratio < 20:
        quality_score += 20
    elif 10 <= rem_ratio < 15:
        quality_score += 10

    if sleep_efficiency >= 90:
        quality_score += 40
    elif 85 <= sleep_efficiency < 90:
        quality_score += 30
    elif 80 <= sleep_efficiency < 85:
        quality_score += 20
    elif 70 <= sleep_efficiency < 80:
        quality_score += 10

    # 睡眠质量等级
    if quality_score >= 90:
        quality_level = "优秀"
    elif quality_score >= 75:
        quality_level = "良好"
    elif quality_score >= 60:
        quality_level = "一般"
    else:
        quality_level = "较差"

    sleep_quality = {
        "subject_id": subject_id,
        "total_frames": len(combined_results),
        "total_sleep_time_min": round(total_sleep_time / 60, 2),
        "total_record_time_min": round(total_record_time / 60, 2),
        "sleep_efficiency_percent": round(sleep_efficiency, 2),
        "deep_sleep_ratio_percent": round(deep_sleep_ratio, 2),
        "light_sleep_ratio_percent": round(light_sleep_ratio, 2),
        "rem_sleep_ratio_percent": round(rem_ratio, 2),
        "stage_transitions": stage_transitions,
        "quality_score": quality_score,
        "quality_level": quality_level,
        "label_stats": label_stats,
        "combined_results": combined_results
    }

    return sleep_quality


# ====================== 第六步：可视化输出 ======================

def visualize_sleep_analysis(sleep_quality, subject_id):
    """
    生成睡眠质量分析的可视化图表（基于整晚数据）
    """
    if sleep_quality is None:
        return

    results_df = sleep_quality["combined_results"]

    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 子图1：整晚睡眠结构图（睡眠分期时间轴）
    ax1 = fig.add_subplot(gs[0, :])

    # 定义睡眠期颜色
    stage_colors = {"R": "#FF6B6B", "1": "#4ECDC4", "2": "#45B7D1", "3": "#6C5CE7"}
    stage_names = {"R": "REM期", "1": "N1期", "2": "N2期", "3": "N3期"}

    labels = results_df["label"].values
    y_positions = []
    y_colors = []

    for label in labels:
        if label in stage_colors:
            y_positions.append(1)
            y_colors.append(stage_colors[label])
        else:
            y_positions.append(0)
            y_colors.append("#BDC3C7")  # 其他期用灰色

    # 绘制睡眠分期图
    x_range = range(len(labels))
    ax1.scatter(x_range, y_positions, c=y_colors, s=10, alpha=0.8)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks([1])
    ax1.set_yticklabels(["睡眠分期"])
    ax1.set_xlabel("时间（30秒/帧）")
    ax1.set_title(
        f"受试者 {subject_id} 整晚睡眠结构图（{sleep_quality['total_frames']}帧，{sleep_quality['total_record_time_min']:.1f}分钟）",
        fontsize=14, fontweight='bold')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=stage_colors[key], label=stage_names[key])
                       for key in stage_colors.keys()]
    ax1.legend(handles=legend_elements, loc='upper right')

    # 子图2：睡眠期时长分布柱状图
    ax2 = fig.add_subplot(gs[1, 0])
    label_stats = sleep_quality["label_stats"]
    sleep_stages = label_stats[label_stats["睡眠期"].isin(["R", "1", "2", "3"])]

    x = sleep_stages["睡眠期"].values
    y = sleep_stages["总时长(秒)"].values / 60  # 转换为分钟
    colors = [stage_colors.get(stage, "#BDC3C7") for stage in x]

    bars = ax2.bar(x, y, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel("睡眠分期")
    ax2.set_ylabel("时长（分钟）")
    ax2.set_title("各睡眠期时长分布（整晚）", fontsize=12, fontweight='bold')

    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=9)

    # 子图3：睡眠期占比饼图
    ax3 = fig.add_subplot(gs[1, 1])

    sizes = sleep_stages["总时长(秒)"].values
    labels_pie = [
        f"{stage_names.get(stage, stage)}\n({sleep_stages[sleep_stages['睡眠期'] == stage]['时长占比(%)'].values[0]:.1f}%)"
        for stage in sleep_stages["睡眠期"].values]
    colors_pie = [stage_colors.get(stage, "#BDC3C7") for stage in sleep_stages["睡眠期"].values]

    ax3.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='',
            startangle=90, textprops={'fontsize': 10})
    ax3.set_title("睡眠期占比（整晚）", fontsize=12, fontweight='bold')

    # 子图4：睡眠质量指标雷达图
    ax4 = fig.add_subplot(gs[2, 0], projection='polar')

    # 归一化各项指标到0-1
    categories = ['深睡眠\n比例', '浅睡眠\n比例', 'REM\n比例', '睡眠\n效率']
    values = [
        min(sleep_quality["deep_sleep_ratio_percent"] / 25, 1.0),  # 理想值25%
        min(sleep_quality["light_sleep_ratio_percent"] / 55, 1.0),  # 理想值55%
        min(sleep_quality["rem_sleep_ratio_percent"] / 25, 1.0),  # 理想值25%
        sleep_quality["sleep_efficiency_percent"] / 100  # 理想值100%
    ]

    # 闭合图形
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax4.plot(angles, values, 'o-', linewidth=2, color='#6C5CE7')
    ax4.fill(angles, values, alpha=0.25, color='#6C5CE7')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax4.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    ax4.set_title("睡眠质量指标（整晚）", fontsize=12, fontweight='bold', pad=20)

    # 子图5：睡眠质量综合评分
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # 综合信息展示
    info_text = f"""
    受试者：{subject_id}

    整晚睡眠质量评估

    总睡眠时长：{sleep_quality['total_sleep_time_min']:.1f} 分钟
    总记录时长：{sleep_quality['total_record_time_min']:.1f} 分钟
    睡眠效率：{sleep_quality['sleep_efficiency_percent']:.1f}%

    深睡眠比例：{sleep_quality['deep_sleep_ratio_percent']:.1f}%
    浅睡眠比例：{sleep_quality['light_sleep_ratio_percent']:.1f}%
    REM睡眠比例：{sleep_quality['rem_sleep_ratio_percent']:.1f}%

    睡眠期转换次数：{sleep_quality['stage_transitions']}
    平均每小时转换：{sleep_quality['stage_transitions'] / (sleep_quality['total_record_time_min'] / 60):.1f}次

    综合评分：{sleep_quality['quality_score']}/100
    睡眠质量：{sleep_quality['quality_level']}
    """

    # 根据质量等级设置颜色
    quality_colors = {
        "优秀": "#00C851",
        "良好": "#33B5E5",
        "一般": "#FFBB33",
        "较差": "#FF4444"
    }
    quality_color = quality_colors.get(sleep_quality['quality_level'], "#666666")

    ax5.text(0.5, 0.5, info_text, fontsize=11, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 在底部添加质量等级标签
    ax5.text(0.5, 0.05, f"★ {sleep_quality['quality_level']} ★",
             fontsize=16, fontweight='bold', ha='center', va='bottom',
             color=quality_color)

    plt.suptitle(f"整晚睡眠质量分析报告 - {subject_id}", fontsize=16, fontweight='bold', y=0.995)

    # 保存图像
    viz_filename = f"{subject_id}_sleep_analysis.png"
    viz_path = os.path.join(results_path, viz_filename)
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f" 可视化图表已保存：{viz_filename}")

    return viz_path


# ====================== 第七步：批量处理测试集 ======================

def process_all_test_files():
    """
    批量处理所有测试文件，按受试者分组分析
    """
    subject_files = get_test_files_grouped()

    if len(subject_files) == 0:
        print("❌ 没有找到测试文件！")
        return

    print(f" 开始处理 {len(subject_files)} 个受试者的数据（每个受试者5个Part）...")
    print("=" * 60)

    all_sleep_quality = []

    for subject_idx, subject_id in enumerate(subject_files.keys(), 1):
        print(f"\n{'=' * 60}")
        print(f"【{subject_idx}/{len(subject_files)}】处理受试者：{subject_id}")
        print(f"{'=' * 60}")

        part_files = subject_files[subject_id]

        # 对每个Part进行预测
        all_part_results = []
        part_frame_counts = []

        for part_idx, part_file in enumerate(part_files, 1):
            print(f"\n  处理 Part {part_idx}/5: {part_file}")

            results_df, _, frame_count = predict_part_file(part_file)

            if results_df is None:
                print(f"   Part {part_idx} 预测失败，跳过")
                continue

            print(f"  Part {part_idx} 预测完成，共 {frame_count} 帧")
            print(f"     标签分布：{results_df['label'].value_counts().to_dict()}")

            all_part_results.append(results_df)
            part_frame_counts.append(frame_count)

        if not all_part_results:
            print(f"  ❌ {subject_id} 所有Part预测均失败，跳过")
            continue

        print(f"\n 合并所有Part进行整晚睡眠分析...")
        print(f"   Part帧数分布：{part_frame_counts}")
        print(f"   总帧数：{sum(part_frame_counts)}")

        # 基于合并的整晚数据分析睡眠质量
        sleep_quality = analyze_sleep_quality(all_part_results, subject_id)

        if sleep_quality:
            all_sleep_quality.append(sleep_quality)

            print(f"\n 整晚睡眠质量结果：")
            print(f"   总睡眠时长：{sleep_quality['total_sleep_time_min']:.1f} 分钟")
            print(f"   总记录时长：{sleep_quality['total_record_time_min']:.1f} 分钟")
            print(f"   睡眠效率：{sleep_quality['sleep_efficiency_percent']:.1f}%")
            print(f"   深睡眠比例：{sleep_quality['deep_sleep_ratio_percent']:.1f}%")
            print(f"   浅睡眠比例：{sleep_quality['light_sleep_ratio_percent']:.1f}%")
            print(f"   REM比例：{sleep_quality['rem_sleep_ratio_percent']:.1f}%")
            print(f"   睡眠期转换次数：{sleep_quality['stage_transitions']}")
            print(f"   睡眠质量：{sleep_quality['quality_level']}（{sleep_quality['quality_score']}/100）")

            # 生成整晚可视化
            visualize_sleep_analysis(sleep_quality, subject_id)

        # 保存每个Part的预测结果（按竞赛要求格式）
        print(f"\n保存各Part预测结果...")
        for part_idx, (part_file, part_result) in enumerate(zip(part_files, all_part_results), 1):
            # 提取Part编号
            part_match = re.search(r'Part[_ ]*(\d+)', part_file)
            if part_match:
                part_num = part_match.group(1)
            else:
                part_num = str(part_idx)

            # 构建结果文件名
            result_filename = f"{subject_id}_Hypnogram_Data_Part_{part_num}_of_5.txt"
            result_path = os.path.join(results_path, result_filename)

            # 格式化输出：start_time end_time duration label
            with open(result_path, 'w') as f:
                for _, row in part_result.iterrows():
                    line = f"{row['start_time']:.1f}\t{row['end_time']:.1f}\t{row['duration']:.1f}\t{row['label']}\n"
                    f.write(line)

            print(f"   Part {part_num} 结果已保存：{result_filename}")

    print("\n" + "=" * 60)
    print(f" 所有受试者处理完成！共处理 {len(all_sleep_quality)} 个受试者")
    print("=" * 60)

    # 生成综合报告
    if all_sleep_quality:
        generate_summary_report(all_sleep_quality)


def generate_summary_report(all_sleep_quality):
    """
    生成所有受试者的睡眠质量综合报告
    """
    # 提取关键指标
    summary_data = []
    for sq in all_sleep_quality:
        summary_data.append({
            "受试者ID": sq["subject_id"],
            "总帧数": sq["total_frames"],
            "总睡眠时长(分钟)": sq["total_sleep_time_min"],
            "总记录时长(分钟)": sq["total_record_time_min"],
            "睡眠效率(%)": sq["sleep_efficiency_percent"],
            "深睡眠比例(%)": sq["deep_sleep_ratio_percent"],
            "浅睡眠比例(%)": sq["light_sleep_ratio_percent"],
            "REM比例(%)": sq["rem_sleep_ratio_percent"],
            "睡眠期转换次数": sq["stage_transitions"],
            "质量评分": sq["quality_score"],
            "质量等级": sq["quality_level"]
        })

    summary_df = pd.DataFrame(summary_data)

    # 保存综合报告
    report_path = os.path.join(results_path, "sleep_quality_summary_report.csv")
    summary_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f" 综合报告已保存：sleep_quality_summary_report.csv")

    # 打印摘要
    print("\n" + "=" * 60)
    print("整晚睡眠质量分析摘要")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1：睡眠效率对比
    ax1 = axes[0, 0]
    ax1.bar(summary_df["受试者ID"], summary_df["睡眠效率(%)"], color='skyblue', alpha=0.8)
    ax1.set_xlabel("受试者ID")
    ax1.set_ylabel("睡眠效率 (%)")
    ax1.set_title("各受试者睡眠效率对比（整晚）", fontweight='bold')
    ax1.axhline(y=85, color='r', linestyle='--', label='健康基准(85%)')
    ax1.legend()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 子图2：睡眠结构对比（堆叠柱状图）
    ax2 = axes[0, 1]
    x = np.arange(len(summary_df))
    width = 0.6

    ax2.bar(x, summary_df["深睡眠比例(%)"], width, label='深睡眠(N3)', color='#6C5CE7')
    ax2.bar(x, summary_df["浅睡眠比例(%)"], width, bottom=summary_df["深睡眠比例(%)"],
            label='浅睡眠(N1+N2)', color='#45B7D1')
    ax2.bar(x, summary_df["REM比例(%)"], width,
            bottom=summary_df["深睡眠比例(%)"] + summary_df["浅睡眠比例(%)"],
            label='REM睡眠', color='#FF6B6B')

    ax2.set_xlabel("受试者ID")
    ax2.set_ylabel("时长占比 (%)")
    ax2.set_title("各受试者睡眠结构对比（整晚）", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df["受试者ID"])
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 子图3：睡眠质量评分对比
    ax3 = axes[1, 0]
    colors = ['#00C851' if level == '优秀' else '#33B5E5' if level == '良好'
    else '#FFBB33' if level == '一般' else '#FF4444' for level in summary_df["质量等级"]]
    ax3.bar(summary_df["受试者ID"], summary_df["质量评分"], color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel("受试者ID")
    ax3.set_ylabel("质量评分")
    ax3.set_title("各受试者睡眠质量评分对比（整晚）", fontweight='bold')
    ax3.set_ylim(0, 100)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # 添加图例说明质量等级
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00C851', label='优秀(≥90)'),
        Patch(facecolor='#33B5E5', label='良好(75-89)'),
        Patch(facecolor='#FFBB33', label='一般(60-74)'),
        Patch(facecolor='#FF4444', label='较差(<60)')
    ]
    ax3.legend(handles=legend_elements, loc='lower left')

    # 子图4：睡眠时长对比
    ax4 = axes[1, 1]
    ax4.bar(summary_df["受试者ID"], summary_df["总睡眠时长(分钟)"], color='lightgreen', alpha=0.8)
    ax4.set_xlabel("受试者ID")
    ax4.set_ylabel("睡眠时长 (分钟)")
    ax4.set_title("各受试者总睡眠时长对比（整晚）", fontweight='bold')
    ax4.axhline(y=420, color='r', linestyle='--', label='健康基准(7小时)')
    ax4.legend()
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    summary_viz_path = os.path.join(results_path, "all_subjects_comparison.png")
    plt.savefig(summary_viz_path, dpi=300, bbox_inches='tight')
    print(f"综合对比图已保存：all_subjects_comparison.png")

    plt.show()


# ====================== 主程序入口 ======================

if __name__ == "__main__":
    print("=" * 60)
    print(" 睡眠质量分析系统 - 测试集预测与评估")
    print("按受试者分组，同一受试者的5个Part合并为整晚数据进行分析")
    print("=" * 60)

    # 批量处理测试集
    process_all_test_files()

    print("\n" + "=" * 60)
    print(" 程序运行完成！")
    print(f" 所有结果已保存至：{results_path}")
    print(" 输出文件包括：")
    print("   - 每个Part的预测结果txt文件（按竞赛要求）")
    print("   - 每个受试者的整晚睡眠分析可视化图")
    print("   - 所有受试者的综合报告CSV")
    print("   - 受试者对比分析图")
    print("=" * 60)
