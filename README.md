# Sleep Stage Analysis System

睡眠分期分析桌面应用 - 脑电数据处理与模型训练平台

## 功能特性

- ✅ 支持大容量脑电数据上传（2-3GB）
- ✅ 数据预处理与滤波
- ✅ 随机森林模型训练
- ✅ 睡眠分期预测
- ✅ 脑电波形可视化
- ✅ 睡眠质量分析

## 安装与运行

### 环境要求

- Node.js >= 20.0.0
- pnpm >= 8.0.0
- Python 3.8+ 及以下依赖包:
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - joblib

### 安装依赖

```bash
pnpm install
```

### 开发模式

```bash
pnpm dev
```

### 构建与打包

```bash
# 构建生产版本
pnpm build

# 打包 Windows 安装包
pnpm package:win

# 打包 macOS 版本
pnpm package:mac

# 打包 Linux 版本
pnpm package:linux
```

## 使用说明

### 1. 准备训练数据

将脑电数据组织为以下结构：
```
train_data/
├── Subject1_EEGFpz_Cz_Part1 of 5.txt
├── Subject1_Hypnogram_Part1 of 5.txt
├── Subject1_EEGFpz_Cz_Part2 of 5.txt
├── Subject1_Hypnogram_Part2 of 5.txt
└── ...
```

### 2. 数据预处理

- 点击"选择"按钮选择训练数据目录
- 点击"开始预处理"进行数据预处理

### 3. 模型训练

- 预处理完成后，点击"开始训练"
- 在"运行日志"标签页查看训练进度

### 4. 模型预测

- 点击"选择"按钮选择测试数据目录
- 点击"开始预测"进行预测
- 在"预测结果"标签页查看结果

### 5. 波形查看

- 在"波形图"标签页查看脑电波形
- 从下拉列表选择文件或点击"随机选择"

## 数据格式

- **脑电数据文件**: 纯文本格式，每行一个采样点，采样率 100Hz
- **标签文件**: 纯文本格式，包含 start_time, end_time, duration, label 列
- **标签类型**: R (REM), 1 (N1), 2 (N2), 3 (N3)

## 技术架构

- **前端**: Electron + Vite + TypeScript + Chart.js
- **后端**: Python (scikit-learn, numpy, pandas, scipy)
- **打包**: electron-builder

## 许可证

MIT
