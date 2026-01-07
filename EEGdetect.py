import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import SeizureCNN

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ================= 配置 =================
MAT_FILE = 'nmm_data.mat'
MODEL_PATH = 'seizure_cnn_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("推理与验证")


def main():

    print(f"正在读取 {MAT_FILE}...")
    mat = scipy.io.loadmat(MAT_FILE)
    raw_data = mat['generated_data'] 
    k_values = mat['K_values'].flatten()


    print(f"数据加载成功: {raw_data.shape}, 共 {len(k_values)} 个 K 值点")

    model = SeizureCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型权重加载成功！")


    # 推理
    confidence_scores = []
    
    print("正在进行判断...")
    with torch.no_grad():
        for i in range(len(k_values)):
            seg = raw_data[i]

            mean = np.mean(seg, axis=1, keepdims=True)
            std = np.std(seg, axis=1, keepdims=True)
            seg_norm = (seg - mean) / (std + 1e-6)
            
            input_tensor = torch.tensor(seg_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            prob = model(input_tensor).item()
            confidence_scores.append(prob)

    # 绘图
    print("正在绘图...")
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, confidence_scores, 'b-o', linewidth=2, markersize=6, label='CNN输出')
    
    # 辅助线
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=60, color='gray', linestyle='--', alpha=0.5, label='转折点')
    
    plt.title('机制验证', fontsize=14)
    plt.xlabel('耦合强度K', fontsize=12)
    plt.ylabel('癫痫信号的概率', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存结果
    plt.savefig('mechanism_result.png', dpi=300)
    plt.show()
    
    print("完成！结果已保存为 mechanism_result.png")
    
    # 简单的自动结论输出
    avg_low_k = np.mean(confidence_scores[:5]) # K 较小时的平均概率
    avg_high_k = np.mean(confidence_scores[-5:]) # K 较大时的平均概率
    print("\n=== 自动分析结论 ===")
    print(f"低耦合强度 (K={k_values[0]}) 时的发作概率: {avg_low_k:.4f}")
    print(f"高耦合强度 (K={k_values[-1]}) 时的发作概率: {avg_high_k:.4f}")
    

if __name__ == "__main__":
    main()