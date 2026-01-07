import os
import glob
import random
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import interp1d

# 配置
TRAIN_FOLDER = './processed_data/'
TEST_MAT = './test/chb01_03.mat'
OUTPUT_FILE = 'cnn_prediction.dat'
MODEL_SAVE_PATH = 'seizure_cnn_model.pth'

TARGET_LEN = 921600

WINDOW_SIZE = 2560
STRIDE = 128
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"深度学习模型")

# 模型
class SeizureCNN(nn.Module):
    def __init__(self):
        super(SeizureCNN, self).__init__()
        # 输入: [Batch, 23, 2560]

        self.layer1 = nn.Sequential(
            nn.Conv1d(23, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 数据准备
def load_training_data():
    print("正在准备训练数据...")
    train_files = glob.glob(os.path.join(TRAIN_FOLDER, '*.mat'))

    pos_samples = []
    neg_samples = []

    for fpath in train_files:
        if 'chb01_03.mat' in fpath: continue

        try:
            mat = scipy.io.loadmat(fpath)
            data = mat['data']
            eeg = data[:, :23]
            labels = data[:, 23]
        except:
            continue

        total_len = eeg.shape[0]
        if total_len < WINDOW_SIZE: continue

        # 发作样本
        seizure_idxs = np.where(labels == 1)[0]
        if len(seizure_idxs) > WINDOW_SIZE:
            start, end = seizure_idxs[0], seizure_idxs[-1]
            for s in range(start, end - WINDOW_SIZE, STRIDE):
                seg = eeg[s: s + WINDOW_SIZE, :].T
                seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)
                pos_samples.append(seg)

        # 正常样本
        non_seizure_idxs = np.where(labels == 0)[0]
        valid_ns = non_seizure_idxs[non_seizure_idxs < (total_len - WINDOW_SIZE)]

        if len(valid_ns) > 0 and len(pos_samples) > 0:
            indices = np.random.choice(valid_ns, size=min(len(valid_ns), 20), replace=False)
            for idx in indices:
                seg = eeg[idx: idx + WINDOW_SIZE, :].T
                seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)
                neg_samples.append(seg)

    # 均衡
    n_pos = len(pos_samples)
    if n_pos == 0:
        print("警告: 未找到发作样本！")
        return torch.randn(10, 23, WINDOW_SIZE), torch.zeros(10, 1)

    if len(neg_samples) > n_pos:
        neg_samples = random.sample(neg_samples, n_pos)

    X = np.array(pos_samples + neg_samples, dtype=np.float32)
    y = np.array([1] * len(pos_samples) + [0] * len(neg_samples), dtype=np.float32).reshape(-1, 1)

    print(f"数据集构建完成: 正样本 {len(pos_samples)} | 负样本 {len(neg_samples)}")
    return torch.tensor(X), torch.tensor(y)

def main():
    X_train, y_train = load_training_data()
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SeizureCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("\n开始训练")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {total_loss / len(dataloader):.4f}")

    print(f"保存模型至 {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n开始推理")
    model.eval()

    mat = scipy.io.loadmat(TEST_MAT)
    test_eeg = mat['data'][:, :23]
    total_samples = test_eeg.shape[0]

    pred_probs = []
    pred_times = []

    print("正在逐窗扫描")
    with torch.no_grad():
        for s in range(0, total_samples - WINDOW_SIZE, STRIDE):
            seg = test_eeg[s: s + WINDOW_SIZE, :].T
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)
            inp = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            prob = model(inp).item()
            pred_probs.append(prob)
            pred_times.append(s)

    print("正在插值对齐")
    x_original = np.array(pred_times)
    y_original = np.array(pred_probs)
    x_target = np.arange(1, TARGET_LEN + 1)

    interp_func = interp1d(x_original, y_original, kind='linear', fill_value="extrapolate")
    y_final = interp_func(x_target)
    y_final = np.clip(y_final, 0, 1)

    print(f"保存至 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write("CNN_Probability\n")
        for val in y_final:
            f.write(f"{val:.6f}\n")

    print("全部完成")


if __name__ == "__main__":
    main()