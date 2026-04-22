# Sleep Stage Analysis System

"生命奇点"参加第七届江苏省生医工比赛的可视化应用
### 若欲在ide中直接运行，请使用 /code 中的代码

### 可以直接在release里下载installer.exe或者直接使用项目文件夹的exe（windows版本）
### 其他os下可参照下面的指导手动打包
## 手动打包方法
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

### 构建与打包

```bash
pnpm build

# Windows 安装包
pnpm package:win

# macOS 版本
pnpm package:mac

#  Linux 版本
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

