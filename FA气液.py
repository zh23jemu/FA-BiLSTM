# -*- coding: utf-8 -*-
"""
渤海深层潜山全井段流体智能评价系统 (保留地质波动细节版)
核心：提取底层连续线性组合值 (Logits)，还原类似于 LDA 的连续波动测井曲线
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# 0. 全局设置与超参数
# ==========================================
WINDOW_SIZE = 10
EPOCHS = 150
LR = 0.01


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)

# ==========================================
# 1. 加载数据并进行全井段预处理
# ==========================================
print(">>> [1/5] 正在加载全井段测井数据...")
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '6-3训练数据_副本3.xlsx')
output_path = os.path.join(current_dir, 'FA_BiLSTM_连续波动曲线_全表.xlsx')

df = pd.read_excel(file_path)

# 使用 4 个抗干扰的高灵敏度参数
features = ['Sigma', 'RICX', 'RIN13', 'RATO13']

# 填补缺失值保证序列连续性
df[features] = df[features].ffill().bfill()
X_all = df[features].values

# 全井段统一标准化
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# ==========================================
# 2. 构建深度序列滑动窗口
# ==========================================
print(">>> [2/5] 正在构建时序深度滑动窗口...")
X_seq_all = []
for i in range(len(X_all_scaled) - WINDOW_SIZE + 1):
    X_seq_all.append(X_all_scaled[i: i + WINDOW_SIZE])
X_seq_all = np.array(X_seq_all)

# 提取有标签段进行训练 (2=气层->0, 3=含液层->1)
valid_seq_indices = []
y_train_list = []

for i in range(len(X_seq_all)):
    target_idx = i + WINDOW_SIZE - 1
    label = df.iloc[target_idx]['label2']
    if pd.notna(label) and label in [2, 3]:
        valid_seq_indices.append(i)
        y_train_list.append(0 if label == 2 else 1)

X_train_seq = X_seq_all[valid_seq_indices]
y_train_seq = np.array(y_train_list)

X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.long)

class_counts = np.bincount(y_train_seq)
weights = len(y_train_seq) / (len(class_counts) * class_counts)
class_weights = torch.tensor(weights, dtype=torch.float32)


# ==========================================
# 3. 定义网络 (FA-BiLSTM)
# ==========================================
class FeatureAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_features, attention_dim=32):
        super().__init__()
        self.W_h = nn.Linear(hidden_size * 2, attention_dim, bias=False)
        self.W_x = nn.Linear(1, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=True)

    def forward(self, h_context, x_current):
        h_proj = self.W_h(h_context).unsqueeze(1)
        x_proj = self.W_x(x_current.unsqueeze(-1))
        energy = torch.tanh(h_proj + x_proj)
        score = self.v(energy).squeeze(-1)
        alpha = F.softmax(score, dim=1)
        weighted_x = alpha * x_current
        return alpha, weighted_x


class FA_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fa = FeatureAttentionLayer(hidden_size, input_size, 32)
        self.fc = nn.Linear(hidden_size * 2 + input_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        h_target = out[:, -1, :]
        x_target = x[:, -1, :]
        alpha, weighted_x = self.fa(h_target, x_target)
        fused = torch.cat([h_target, weighted_x], dim=1)
        # 返回未被 Softmax 饱和的连续值 (Logits)
        return self.fc(fused), alpha


# ==========================================
# 4. 训练网络
# ==========================================
print(f">>> [3/5] 正在基于已知标签段进行网络训练...")
model = FA_BiLSTM(input_size=len(features), hidden_size=16, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(weight=class_weights)

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    model.lstm.flatten_parameters()
    out, _ = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

# ==========================================
# 5. 【核心修复】：提取类似于 LDA 的连续波动曲线
# ==========================================
print(">>> [4/5] 正在面向整口井进行连续波动推演...")
model.eval()
X_all_t = torch.tensor(X_seq_all, dtype=torch.float32)

with torch.no_grad():
    logits, alphas = model(X_all_t)

    # 1. 提取气层的连续组合值 (等同于 LDA 权重乘参数的和)
    gas_logits = logits[:, 0].numpy()

    # 【新增逻辑 1】：将 gas_logits 缩放到 -1 到 1 的范围
    scaler_gas = MinMaxScaler(feature_range=(-1, 1))
    gas_logits_scaled = scaler_gas.fit_transform(gas_logits.reshape(-1, 1)).flatten()

    # 2. 使用“温度标度 (Temperature Scaling)”打破 Softmax 的 0/1 饱和
    temperature = 3.0
    probs_smooth = F.softmax(logits / temperature, dim=1).numpy()

    preds = np.argmax(F.softmax(logits, dim=1).numpy(), axis=1).astype(float)
    alphas_np = alphas.numpy()

pad_len = WINDOW_SIZE - 1

# 映射：0->2(气), 1->3(液)
pred_label_mapped = np.where(np.pad(preds, (pad_len, 0), constant_values=np.nan) == 0, 2,
                             np.where(np.pad(preds, (pad_len, 0), constant_values=np.nan) == 1, 3, np.nan))

# 【专为您输出的两条具有真实物理波动感的曲线】：
# 曲线1：连续气层指示指数 (类似您的 Label2 计算公式)
# 注意：这里使用的是缩放后的 gas_logits_scaled
gas_index_curve = np.pad(gas_logits_scaled, (pad_len, 0), constant_values=np.nan)
df['FA_Gas_Index'] = gas_index_curve

# 【新增逻辑 2】：将 I_reservoir 列数值小于0的深度对应的 FA_Gas_Index 统一改为0
if 'I_reservoir' in df.columns:
    df.loc[df['I_reservoir'] < 0, 'FA_Gas_Index'] = 0
else:
    print(">>> 警告: 原数据中未发现 'I_reservoir' 列，请检查表头名是否一致！")

# 曲线2：平滑气层概率 (不再是平头方波，而是像测井曲线一样上下波动)
p_gas_smooth_curve = np.pad(probs_smooth[:, 0], (pad_len, 0), constant_values=np.nan)
df['FA_p_gas_smooth'] = p_gas_smooth_curve

# 预测结论
df['FA_pred_label2'] = pred_label_mapped

# 保存动态权重曲线
for i, feat in enumerate(features):
    alpha_curve = np.pad(alphas_np[:, i], (pad_len, 0), constant_values=np.nan)
    df[f'FA_alpha_{feat}'] = alpha_curve

print(">>> [5/5] 正在保存结果 Excel...")
df.to_excel(output_path, index=False)
print(f"\n✅ 全部完成！包含真实波动细节的曲线已保存至：{output_path}")