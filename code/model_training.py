"""
该文件基于预处理后的npy文件完成：
1. 读取30秒脑电帧和对应标签
2. 提取时域+频域基础特征
3. 标签编码（将R/1/2/3转为数字，适配模型输入）
4. 训练随机森林模型，计算宏F1值
5. 保存模型和特征数据，方便后续预测
"""

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')  # 忽略无关警告
# ====================== 第一步：配置路径 + 读取预处理数据 ======================
# 替换为你保存npy文件的路径（和上一步的save_path一致）
processed_data_path = r"..\data\processed_data"

# 读取npy文件
frames_path = os.path.join(processed_data_path, "all_eeg_frames.npy")
labels_path = os.path.join(processed_data_path, "all_frame_labels.npy")
try:
    eeg_frames = np.load(frames_path)
    frame_labels = np.load(labels_path)
    print(f"✅ 成功读取预处理数据：")
    print(f"   - 脑电帧形状：{eeg_frames.shape} (帧数 × 每帧数据点)")
    print(f"   - 标签数量：{len(frame_labels)}")
    print(f"   - 标签分布：{pd.Series(frame_labels).value_counts().to_dict()}")
    print("=" * 60)

except FileNotFoundError:
    print(f"❌ 未找到npy文件，请检查路径：{processed_data_path}")
    print(f"   确认上一步预处理已成功保存 all_eeg_frames.npy 和 all_frame_labels.npy")
    exit()

# ====================== 第二步：特征提取（时域+频域，核心环节） ======================
def extract_eeg_features(frame, fs=100):
    """
    提取单帧脑电数据的特征（30秒/3000个点）
    特征说明：
    - 时域特征：描述信号的统计特性（均值、标准差、最大值、最小值等）
    - 频域特征：描述信号的频率分布（不同频段能量占比）
    """
    # ---------------------- 2.1 基础时域特征 ----------------------
    mean_val = np.mean(frame)  # 均值
    std_val = np.std(frame)  # 标准差
    max_val = np.max(frame)  # 最大值
    min_val = np.min(frame)  # 最小值
    ptp_val = np.ptp(frame)  # 峰峰值（最大值-最小值）
    rms_val = np.sqrt(np.mean(np.square(frame)))  # 均方根
    skewness_val = stats.skew(frame)  # 偏度（分布对称性）
    kurtosis_val = stats.kurtosis(frame)  # 峰度（分布陡峭程度）
    zero_cross = np.sum(np.diff(np.sign(frame)) != 0)  # 过零率（信号穿越0点次数）
    # ---------------------- 2.2 新增时域特征（增强区分度） ----------------------
    abs_frame = np.abs(frame)
    avg_abs = np.mean(abs_frame)  # 平均绝对值
    peak_factor = max_val / rms_val if rms_val != 0 else 0  # 峰值因子（对冲击信号敏感）
    waveform_factor = rms_val / avg_abs if avg_abs != 0 else 0  # 波形因子
    impulse_factor = max_val / avg_abs if avg_abs != 0 else 0  # 脉冲因子
    crest_factor = max_val / np.sqrt(np.mean(np.square(frame))) if rms_val != 0 else 0  # 波峰因子
    variance_val = np.var(frame)  # 方差
    median_val = np.median(frame)  # 中位数
    q25_val = np.percentile(frame, 25)  # 25分位数
    q75_val = np.percentile(frame, 75)  # 75分位数
    iqr_val = q75_val - q25_val  # 四分位距
    # ---------------------- 2.3 频域特征（基于FFT） ----------------------
    # 快速傅里叶变换，获取频率和功率
    n = len(frame)
    freq = np.fft.fftfreq(n, 1 / fs)  # 频率轴
    fft_vals = np.fft.fft(frame)  # FFT结果
    power = np.abs(fft_vals) ** 2  # 功率谱
    # 只保留正频率部分
    pos_mask = freq > 0
    freq_pos = freq[pos_mask]
    power_pos = power[pos_mask]
    # 定义睡眠相关频段
    bands = {
        "delta": (0.5, 4),  # δ波：0.5-4Hz（深睡眠）
        "theta": (4, 8),  # θ波：4-8Hz（浅睡眠）
        "alpha": (8, 13),  # α波：8-13Hz（清醒/浅睡）
        "beta": (13, 30)  # β波：13-30Hz（REM睡眠）
    }

    # 计算各频段能量占比 + 新增频段统计特征
    band_energy = {}
    band_mean = {}  # 各频段功率均值
    band_std = {}  # 各频段功率标准差
    total_energy = np.sum(power_pos)
    if total_energy == 0:
        total_energy = 1e-6  # 避免除以0
    for band_name, (low, high) in bands.items():
        # 筛选该频段的频率索引
        band_mask = (freq_pos >= low) & (freq_pos <= high)
        band_power = power_pos[band_mask]
        # 计算该频段能量占比
        band_energy[band_name] = np.sum(band_power) / total_energy
        # 新增：频段功率均值和标准差
        band_mean[band_name] = np.mean(band_power) if len(band_power) > 0 else 0
        band_std[band_name] = np.std(band_power) if len(band_power) > 0 else 0

    # 新增：谱熵（反映频率分布均匀性，REM睡眠谱熵更高）

    normalized_power = power_pos / total_energy
    spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))  # 加小值避免log(0)

    # 新增：总功率
    total_power = np.sum(power_pos)

    # ---------------------- 2.4 整合所有特征（原特征+新增特征） ----------------------
    features = [
        # 基础时域特征（9个）
        mean_val, std_val, max_val, min_val, ptp_val, rms_val,
        skewness_val, kurtosis_val, zero_cross,
        # 新增时域特征（10个）
        avg_abs, peak_factor, waveform_factor, impulse_factor, crest_factor,
        variance_val, median_val, q25_val, q75_val, iqr_val,

        # 基础频域特征（4个）
        band_energy["delta"], band_energy["theta"],
        band_energy["alpha"], band_energy["beta"],

        # 新增频域特征（9个）
        band_mean["delta"], band_mean["theta"], band_mean["alpha"], band_mean["beta"],
        band_std["delta"], band_std["theta"], band_std["alpha"], band_std["beta"],
        spectral_entropy, total_power
    ]

    return np.array(features)

# 定义少数类数据增强函数（仅对R类添加轻微高斯噪声）
def augment_minority_class(frames, labels, noise_std=0.05):
    """
    对少数类（R类）样本进行数据增强：添加高斯噪声
    """
    # 分离R类和非R类样本
    r_mask = labels == 'R'
    r_frames = frames[r_mask]
    non_r_frames = frames[~r_mask]
    non_r_labels = labels[~r_mask]

    # 对R类样本添加高斯噪声（生成2倍于原数量的增强样本）
    augmented_r_frames = []
    for frame in r_frames:
        noise = np.random.normal(0, noise_std * np.std(frame), size=frame.shape)
        augmented_frame = frame + noise
        augmented_r_frames.append(augmented_frame)
        # 再添加一个噪声强度稍低的增强样本
        noise2 = np.random.normal(0, 0.7 * noise_std * np.std(frame), size=frame.shape)
        augmented_frame2 = frame + noise2
        augmented_r_frames.append(augmented_frame2)

    # 合并原始非R类和增强后的R类
    augmented_frames = np.concatenate([non_r_frames, np.array(augmented_r_frames)])
    augmented_labels = np.concatenate([non_r_labels, np.array(['R'] * len(augmented_r_frames))])

    return augmented_frames, augmented_labels


# 批量提取所有帧的特征
print("🔄 开始提取特征（时域+频域）...")

# 第一步：对少数类进行数据增强
print("🔄 对少数类（R类）进行数据增强...")
eeg_frames_aug, frame_labels_aug = augment_minority_class(eeg_frames, frame_labels)
print(
    f"✅ 数据增强完成：增强后总帧数={len(eeg_frames_aug)}, 标签分布={pd.Series(frame_labels_aug).value_counts().to_dict()}")

# 第二步：提取增强后数据的特征
feature_list = []
for i, frame in enumerate(eeg_frames_aug):
    # 每处理1000帧打印进度
    if i % 1000 == 0 and i > 0:
        print(f"   已处理 {i}/{len(eeg_frames_aug)} 帧")
    features = extract_eeg_features(frame)
    feature_list.append(features)

# 转为特征矩阵（行数=帧数，列数=特征数）

features_matrix = np.array(feature_list)

feature_names = [
    # 基础时域特征（9个）
    "mean", "std", "max", "min", "ptp", "rms",
    "skewness", "kurtosis", "zero_cross",
    # 新增时域特征（10个）
    "avg_abs", "peak_factor", "waveform_factor", "impulse_factor", "crest_factor",
    "variance", "median", "q25", "q75", "iqr",
    # 基础频域特征（4个）
    "delta_energy", "theta_energy", "alpha_energy", "beta_energy",
    # 新增频域特征（9个）
    "delta_mean", "theta_mean", "alpha_mean", "beta_mean",
    "delta_std", "theta_std", "alpha_std", "beta_std",
    "spectral_entropy", "total_power"
]

print(f"✅ 特征提取完成：")
print(f"   - 特征矩阵形状：{features_matrix.shape} (帧数 × 特征数)")
print(f"   - 特征总数：{len(feature_names)}")
print("=" * 60)

# ====================== 第三步：数据预处理（标签编码 + 数据集划分） ======================
# 3.1 标签编码：将R/1/2/3转为数字（模型只能处理数值输入）
le = LabelEncoder()
labels_encoded = le.fit_transform(frame_labels_aug)
print(f"✅ 标签编码完成：")
print(f"   - 编码映射：{dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"   - 编码后标签分布：{pd.Series(labels_encoded).value_counts().to_dict()}")
# 3.2 划分训练集和测试集（7:3分割，保证数据分布一致）

X_train, X_test, y_train, y_test = train_test_split(
    features_matrix, labels_encoded,
    test_size=0.3, random_state=42,  # random_state固定，结果可复现
    stratify=labels_encoded  # 分层抽样，保证训练/测试集标签分布一致
)

print(f"✅ 数据集划分完成：")
print(f"   - 训练集：{X_train.shape} (样本数 × 特征数)")
print(f"   - 测试集：{X_test.shape} (样本数 × 特征数)")
print("=" * 60)

# ====================== 第四步：模型训练（随机森林，优化参数+类别平衡） ======================
# 初始化随机森林分类器（优化参数+类别权重平衡）
rf_model = RandomForestClassifier(
    n_estimators=1550,  # 增加决策树数量，提升稳定性
    max_depth=40,  # 适当增加深度，捕捉复杂特征交互
    min_samples_split=5,  # 最小分裂样本数，避免过拟合
    min_samples_leaf=2,  # 最小叶子节点样本数，避免过拟合
    max_features='sqrt',  # 使用平方根特征数，减少特征相关性影响
    class_weight='balanced_subsample',  # 按子样本平衡类别权重，适配少数类
    random_state=42,  # 随机种子，结果可复现
    n_jobs=-1,  # 使用所有CPU核心，加速训练
    bootstrap=True,
    oob_score=True  # 启用袋外得分，评估模型泛化能力
)

# 训练模型
print("🔄 开始训练随机森林模型...")
rf_model.fit(X_train, y_train)
print(f"✅ 模型训练完成！袋外得分：{rf_model.oob_score_:.4f}")
print("=" * 60)

# ====================== 第五步：模型评估（核心指标：宏F1值） ======================
# 5.1 预测测试集
y_pred = rf_model.predict(X_test)
# 5.2 计算核心指标（宏F1值）
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"📊 模型评估结果（核心指标）：")
print(f"   - 宏F1值：{macro_f1:.4f} (竞赛核心评价指标)")
print(f"   - 加权F1值：{weighted_f1:.4f}")
print("-" * 40)

# 5.3 详细分类报告（精确率、召回率、F1值）

print(f"📋 详细分类报告：")
print(classification_report(
    y_test, y_pred,
    target_names=[f"{cls}({le.inverse_transform([idx])[0]})" for idx, cls in enumerate(le.classes_)]
))

# 5.4 混淆矩阵（直观展示分类效果）
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"真实_{cls}" for cls in le.classes_],
    columns=[f"预测_{cls}" for cls in le.classes_]
)

print(f"🔍 混淆矩阵：")
print(cm_df)
print("=" * 60)

# ====================== 第六步：保存模型和特征数据（方便后续预测/优化） ======================

# 创建保存目录
save_model_path = os.path.join(processed_data_path, "model")
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
# 保存模型（使用joblib，比pickle更适合 sklearn 模型）
import joblib
joblib.dump(rf_model, os.path.join(save_model_path, "sleep_stage_rf_model_optimized.pkl"))
joblib.dump(le, os.path.join(save_model_path, "label_encoder_optimized.pkl"))
# 保存特征矩阵和标签

np.save(os.path.join(save_model_path, "features_matrix_optimized.npy"), features_matrix)
np.save(os.path.join(save_model_path, "labels_encoded_optimized.npy"), labels_encoded)
# 保存特征名称（方便后续分析）

with open(os.path.join(save_model_path, "feature_names_optimized.txt"), "w") as f:
    f.write("\n".join(feature_names))

print(f"✅ 优化后的模型和数据已保存至：{save_model_path}")
print(f"   - 模型文件：sleep_stage_rf_model_optimized.pkl")
print(f"   - 标签编码器：label_encoder_optimized.pkl")
print(f"   - 特征矩阵：features_matrix_optimized.npy")
print("=" * 60)

# ====================== 特征重要性分析（优化模型用） ======================

# 计算特征重要性
feature_importance = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance
}).sort_values(by="importance", ascending=False)

print(f"📈 特征重要性排名（前10）：")
print(importance_df.head(10))
# 可视化时域特征重要性
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 设置默认字体防止汉字显示错误
plt.figure(figsize=(12, 8))
plt.barh(importance_df["feature"][:15], importance_df["importance"][:15])
plt.xlabel("重要性得分")
plt.ylabel("特征名称")
plt.title("脑电特征重要性排名（前15）")
plt.tight_layout()
plt.savefig(os.path.join(save_model_path, "feature_importance_optimized.png"))
plt.show()

print("\n 所有优化步骤完成！模型训练+评估+保存全部结束，优化后宏F1值已显示。")
