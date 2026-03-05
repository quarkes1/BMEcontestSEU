"""
该文件基于预处理后的npy文件完成：
1. 读取30秒脑电帧和对应标签
2. 提取时域+频域基础特征
3. 标签编码（将R/1/2/3转为数字，适配模型输入）
4. 训练随机森林模型，计算宏F1值
5. 保存模型和特征数据，方便后续预测
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import warnings

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
    # ---------------------- 2.1 时域特征 ----------------------
    mean_val = np.mean(frame)  # 均值
    std_val = np.std(frame)  # 标准差
    max_val = np.max(frame)  # 最大值
    min_val = np.min(frame)  # 最小值
    ptp_val = np.ptp(frame)  # 峰峰值（最大值-最小值）
    rms_val = np.sqrt(np.mean(np.square(frame)))  # 均方根
    skewness_val = stats.skew(frame)  # 偏度（分布对称性）
    kurtosis_val = stats.kurtosis(frame)  # 峰度（分布陡峭程度）
    zero_cross = np.sum(np.diff(np.sign(frame)) != 0)  # 过零率（信号穿越0点次数）

    # ---------------------- 2.2 频域特征（基于FFT） ----------------------
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
        "alpha": (8, 13),  # α波：8-13Hz（清醒/浅睡眠）
        "beta": (13, 30)  # β波：13-30Hz（REM睡眠）
    }

    # 计算各频段能量占比
    band_energy = {}
    total_energy = np.sum(power_pos)
    if total_energy == 0:
        total_energy = 1e-6  # 避免除以0

    for band_name, (low, high) in bands.items():
        # 筛选该频段的频率索引
        band_mask = (freq_pos >= low) & (freq_pos <= high)
        # 计算该频段能量占比
        band_energy[band_name] = np.sum(power_pos[band_mask]) / total_energy

    # ---------------------- 2.3 整合所有特征 ----------------------
    features = [
        # 时域特征
        mean_val, std_val, max_val, min_val, ptp_val, rms_val,
        skewness_val, kurtosis_val, zero_cross,
        # 频域特征（各频段能量占比）
        band_energy["delta"], band_energy["theta"],
        band_energy["alpha"], band_energy["beta"]
    ]

    return np.array(features)


# 批量提取所有帧的特征
print("🔄 开始提取特征（时域+频域）...")
feature_list = []
for i, frame in enumerate(eeg_frames):
    # 每处理1000帧打印进度
    if i % 1000 == 0 and i > 0:
        print(f"   已处理 {i}/{len(eeg_frames)} 帧")
    features = extract_eeg_features(frame)
    feature_list.append(features)

# 转为特征矩阵（行数=帧数，列数=特征数）
features_matrix = np.array(feature_list)
feature_names = [
    "mean", "std", "max", "min", "ptp", "rms",
    "skewness", "kurtosis", "zero_cross",
    "delta", "theta", "alpha", "beta"
]

print(f"✅ 特征提取完成：")
print(f"   - 特征矩阵形状：{features_matrix.shape} (帧数 × 特征数)")
print(f"   - 特征名称：{feature_names}")
print("=" * 60)

# ====================== 第三步：数据预处理（标签编码 + 数据集划分） ======================
# 3.1 标签编码：将R/1/2/3转为数字（模型只能处理数值输入）
le = LabelEncoder()
labels_encoded = le.fit_transform(frame_labels)
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

# ====================== 第四步：模型训练（随机森林，新手友好+效果稳定） ======================
# 初始化随机森林分类器（参数适配新手，无需调参也能有不错效果）
rf_model = RandomForestClassifier(
    n_estimators=100,  # 决策树数量（100棵足够）
    max_depth=10,  # 树最大深度（避免过拟合）
    random_state=42,  # 随机种子，结果可复现
    n_jobs=-1  # 使用所有CPU核心，加速训练
)

# 训练模型
print("🔄 开始训练随机森林模型...")
rf_model.fit(X_train, y_train)
print("✅ 模型训练完成！")
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

joblib.dump(rf_model, os.path.join(save_model_path, "sleep_stage_rf_model.pkl"))
joblib.dump(le, os.path.join(save_model_path, "label_encoder.pkl"))

# 保存特征矩阵和标签
np.save(os.path.join(save_model_path, "features_matrix.npy"), features_matrix)
np.save(os.path.join(save_model_path, "labels_encoded.npy"), labels_encoded)

# 保存特征名称（方便后续分析）
with open(os.path.join(save_model_path, "feature_names.txt"), "w") as f:
    f.write("\n".join(feature_names))

print(f"✅ 模型和数据已保存至：{save_model_path}")
print(f"   - 模型文件：sleep_stage_rf_model.pkl")
print(f"   - 标签编码器：label_encoder.pkl")
print(f"   - 特征矩阵：features_matrix.npy")
print("=" * 60)

# ====================== 特征重要性分析（优化模型用） ======================
# 计算特征重要性
feature_importance = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance
}).sort_values(by="importance", ascending=False)

print(f"📈 特征重要性排名（前5）：")
print(importance_df.head(5))

# 可视化时域特征重要性

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False #设置默认字体防止汉字显示错误

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"][:10], importance_df["importance"][:10])
plt.xlabel("重要性得分")
plt.ylabel("特征名称")
plt.title("脑电特征重要性排名（前10）")
plt.tight_layout()
plt.savefig(os.path.join(save_model_path, "feature_importance.png"))
plt.show()

print("\n 所有步骤完成！模型训练+评估+保存全部结束，宏F1值已显示。")