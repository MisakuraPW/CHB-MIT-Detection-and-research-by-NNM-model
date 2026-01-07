# 数据-机理融合驱动的癫痫脑电监测与同步化机制研究
# (Data-Mechanism Fusion for Epilepsy Detection and Synchronization Research)

![Matlab](https://img.shields.io/badge/Matlab-R2021a%2B-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red) ![License](https://img.shields.io/badge/License-MIT-blue)

## 📖 项目简介 (Introduction)

本项目针对癫痫自动监测与致病机理探究这一临床痛点，构建了一个**“数据-机理融合”**的闭环研究框架。项目包含两个核心部分：

1.  **数据驱动监测**：基于 **CHB-MIT** 临床脑电数据集，构建了特征工程（线长/熵+SVM）与端到端深度学习（1D-CNN）模型，实现了高精度的癫痫发作检测。
2.  **机理驱动仿真**：基于扩展的 **多通道神经群模型 (Multi-channel Jansen-Rit NMM)**，通过调节耦合强度参数 ($K$) 模拟脑区间的过度同步化行为，并利用预训练的 CNN 模型进行闭环验证，揭示了癫痫发作的动力学机制。

## 🌟 主要特性 (Key Features)

* **双路径检测模型**：对比了传统机器学习 (SVM + Line Length) 与深度学习 (1D-CNN) 的性能，CNN 准确率达到 98% 以上。
* **交互式可视化工具**：基于 MATLAB App Designer 开发的 GUI，支持实时查看脑电波形与线长、熵特征的动态变化。
* **多通道 NMM 仿真**：实现了 23 通道全连接耦合的神经群模型，支持多动力学支路模拟宽带信号。
* **闭环验证 (S-Curve)**：定量绘制了“耦合强度-发作概率”的 S 型曲线，验证了过度同步化机制。

## 🛠️ 环境依赖 (Requirements)

本项目代码分为 MATLAB 和 Python 两部分。

### MATLAB
* MATLAB R2021a 或更高版本
* Signal Processing Toolbox (信号处理工具箱)
* Statistics and Machine Learning Toolbox (统计与机器学习工具箱)

### Python
* Python 3.8+
* PyTorch (建议使用 GPU 版本)
* NumPy, SciPy, Matplotlib


安装 Python 依赖：
```bash
pip install torch numpy scipy matplotlib
```

## 📂 文件功能说明 (File Description)

所有源代码位于项目根目录下，按功能分类如下：

### 1. 数据预处理 (Preprocessing)
* `preprocess_dataset.m`: **[主程序]** 读取 EDF 文件，执行滤波 (0.5-40Hz)、去噪、切片，并将数据与标签保存为 `.mat` 文件。
* `label.m` / `labelEEG.m`: 自动化标注脚本，读取 `.seizures` 文件并生成 0/1 标签向量。
* `get_seizure_period.m`: 辅助函数，解析发作起止时间。

### 2. 特征工程 (Feature Engineering)
* `linelength.m`: 计算线长 (Line Length) 特征（并行计算）。
* `entropy.m`: 计算香农熵、频域熵、样本熵。
* `AR.m`: 计算自回归模型预测误差 (NPE)。
* `AIC.m`: 用于确定 AR 模型的最佳阶数 (AIC准则)。
* `visualize_signal.mlapp`: **[GUI]** 交互式脑电信号与特征可视化软件。

### 3. 深度学习与分类 (DL & Classification)
* `model.py`: 定义 1D-CNN 网络架构 (PyTorch)。
* `train.py`: 深度学习模型训练脚本。
* `ML.m`: 加载预训练 SVM 模型进行推理与概率校准。
* `EEGdetect.py`: **[推理]** 加载训练好的 CNN 模型，对仿真数据进行闭环检测。
* `seizure_cnn_model.pth`: 预训练好的 CNN 模型权重文件。

### 4. 机理仿真 (Mechanism Simulation)
* `NMM.m`: **[核心]** 多通道神经群模型仿真主程序。扫描耦合强度 $K$，生成模拟脑电数据。
* `test_generate.m`: 快速测试脚本，用于验证 NMM 单次生成的波形。

### 5. 其他 (Others)
* `viewEEG.m`: 简单的单通道/多通道波形绘图工具。

## 🚀 快速开始 (Quick Start)

### 第一步：数据准备
请确保 CHB-MIT 数据集文件（`.edf`）位于 `data/` 目录下（需自行下载），然后运行 MATLAB 预处理脚本：
```matlab
% 在 Matlab 中运行
preprocess_dataset
```
### 第二步：模型训练 (CNN)
使用 Python 脚本训练深度学习模型：
```Bash
python train.py
```
训练完成后，模型权重将更新至 seizure_cnn_model.pth。
### 第三步：机理模型仿真
运行 MATLAB 脚本，生成不同耦合强度下的模拟脑电数据：
```Matlab
% 在 Matlab 中运行
NMM
```
该脚本将扫描 $K \in [0, 100]$，并将仿真结果保存为 nmm_data.mat。
### 第四步：闭环验证
利用训练好的 CNN 对仿真数据进行评估，绘制 S 型曲线：
```Bash
python EEGdetect.py
```
