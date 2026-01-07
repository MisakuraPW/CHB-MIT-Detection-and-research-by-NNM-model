import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import glob
import os
import sys
from model import Simple1DCNN
import mat73

# 配置
DATA_DIR = "./processed_data" 
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
WINDOW_SIZE = 512      
STRIDE = 128           
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CHBDataset(Dataset):
    def __init__(self, data_dir, window_size, stride, mode='train'):
        self.window_size = window_size
        self.stride = stride
        self.samples_index = []
        
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if len(all_files) == 0:
            print("错误: 未找到任何 .mat 文件，请检查路径:", data_dir)
            sys.exit(1)

        split_idx = int(len(all_files) * 0.8)
        if mode == 'train':
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
            
        print(f"[{mode}] 正在索引 {len(self.files)} 个文件 (不加载到内存)...")
        
        for f in self.files:
            try:
                try:
                    file_info = scipy.io.whosmat(f)
                    data_shape = None
                    for name, shape, dtype in file_info:
                        if name == 'data':
                            data_shape = shape
                            break
                    
                    if data_shape is None:
                        print(f"警告: 文件 {f} 中没有找到 'data' 变量，跳过。")
                        continue
                        
                    total_len = data_shape[0] # [Time, Channels]
                    
                except Exception:
                    if mat73:
                        data_dict = mat73.loadmat(f, only_include=['data'])
                        if isinstance(data_dict['data'], dict):
                            print(f"警告: 文件 {f} 格式异常，跳过")
                            continue
                        total_len = data_dict['data'].shape[0]
                    else:
                        print(f"警告: 无法读取文件 {f} (可能是v7.3格式且未安装mat73)，跳过。")
                        continue

                if total_len > window_size:
                    n_samples = (total_len - window_size) // stride
                    for i in range(n_samples):
                        start = i * stride
                        self.samples_index.append((f, start))
                        
            except Exception as e:
                print(f"索引错误 {f}: {e}")
        
        print(f"[{mode}] 索引完成，共 {len(self.samples_index)} 个样本。")

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        file_path, start = self.samples_index[idx]
        
        try:
            try:
                mat = scipy.io.loadmat(file_path)
                data = mat['data']
            except:
                if mat73:
                    data = mat73.loadmat(file_path)['data']
                else:
                    raise ValueError("无法读取文件")

            # 截取窗口
            end = start + self.window_size
            window = data[start:end, :]
            
            # 数据处理
            eeg = window[:, :23].T 
            label_seq = window[:, 23]

            label = 1 if np.sum(label_seq) > 0 else 0
            
            return torch.tensor(eeg, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"读取异常 {file_path}: {e}")
            return torch.zeros((23, self.window_size)), torch.tensor(0, dtype=torch.long)

# 训练
def main():
    print(f"使用设备: {DEVICE}")
    
    train_dataset = CHBDataset(DATA_DIR, WINDOW_SIZE, STRIDE, mode='train')
    val_dataset = CHBDataset(DATA_DIR, WINDOW_SIZE, STRIDE, mode='val')
    
    if len(train_dataset) == 0:
        print("错误: 训练集为空！")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 初始化模型
    model = Simple1DCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    best_acc = 0.0
    
    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        batch_count = 0
        
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # 打印进度
            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1} [Batch {i+1}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')
            
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = 100 * correct / total if total > 0 else 0
        avg_loss = train_loss/batch_count if batch_count > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> 最佳模型已保存!")

if __name__ == "__main__":
    main()