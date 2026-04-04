# -*- coding: utf-8 -*-
"""
渤海深层潜山全井段【有效储层】智能识别系统 (归一化至 -1 到 1 波动版)
替代原有的 LDA1.py，输出连续的储层指示指数，并自动将其映射至标准区间 [-1, 1]
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
file_path = os.path.join(current_dir, '6-3训练数据_DT.xlsx')
output_path = os.path.join(current_dir, 'FA_BiLSTM_有效储层识别_全表.xlsx')

df = pd.read_excel(file_path)

# 储层敏感特征
features = ['MLR', 'AMPST', 'PHIE']
label_col = 'label1'

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

valid_seq_indices = []
y_train_list = []

for i in range(len(X_seq_all)):
    target_idx = i + WINDOW_SIZE - 1
    label = df.iloc[target_idx][label_col]
    if pd.notna(label) and label in [0, 1]:
        valid_seq_indices.append(i)
        y_train_list.append(int(label))

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
        return self.fc(fused), alpha


# ==========================================
# 4. 训练网络
# ==========================================
print(f">>> [3/5] 正在基于 {len(y_train_seq)} 个已知标签段进行网络训练...")
model = FA_BiLSTM(input_size=len(features), hidden_size=16, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(weight=class_weights)

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    model.lstm.flatten_parameters()

    noisy_train = X_train_t + torch.randn_like(X_train_t) * 0.3

    out, _ = model(noisy_train)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

# ==========================================
# 5. 全井段推演与 【归一化 (-1 到 1)】
# ==========================================
print(">>> [4/5] 正在面向整口井进行有效储层推演并映射至 [-1, 1] 区间...")
model.eval()
X_all_t = torch.tensor(X_seq_all, dtype=torch.float32)

with torch.no_grad():
    logits, alphas = model(X_all_t)
    # 提取有效储层对应的连续组合值
    res_logits = logits[:, 1].numpy()

    # 平滑概率
    temperature = 3.0
    probs_smooth = F.softmax(logits / temperature, dim=1).numpy()

    # 严格标签分类
    preds = np.argmax(F.softmax(logits, dim=1).numpy(), axis=1).astype(float)
    alphas_np = alphas.numpy()

pad_len = WINDOW_SIZE - 1

# 补齐前 9 个缺失的深度点
res_index_curve = np.pad(res_logits, (pad_len, 0), constant_values=np.nan)

# ========================================================
# 🌟 【新增核心功能】：将连续曲线严格线性归一化到 [-1, 1] 🌟
# ========================================================
# 1. 找到所有非 NaN 的有效数值进行缩放
valid_idx = ~np.isnan(res_index_curve)
valid_res_values = res_index_curve[valid_idx].reshape(-1, 1)

# 2. 调用 MinMaxScaler 强制缩放至 [-1, 1] 区间
scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
norm_res_values = scaler_minmax.fit_transform(valid_res_values).flatten()

# 3. 将缩放后的值填充回包含 NaN 的全井段曲线中
res_index_norm_curve = np.full_like(res_index_curve, np.nan)
res_index_norm_curve[valid_idx] = norm_res_values
# ========================================================

p_res_smooth_curve = np.pad(probs_smooth[:, 1], (pad_len, 0), constant_values=np.nan)
pred_label1_curve = np.pad(preds, (pad_len, 0), constant_values=np.nan)

# ==========================================
# 6. 将结果写回数据表并导出
# ==========================================
# 1. 保存【归一化后】的连续储层指示指数 (-1到1，最适合画图)
df['FA_I_reservoir_Norm'] = res_index_norm_curve

# (可选) 2. 保留原始未缩放的 Logits 值，供对照参考
df['FA_I_reservoir_Raw'] = res_index_curve

# 3. 平滑概率与最终预测标签
df['FA_p_reservoir_smooth'] = p_res_smooth_curve
df['FA_pred_label1'] = pred_label1_curve

# 4. 动态注意力权重
for i, feat in enumerate(features):
    alpha_curve = np.pad(alphas_np[:, i], (pad_len, 0), constant_values=np.nan)
    df[f'FA_alpha_{feat}'] = alpha_curve

print(">>> [5/5] 正在保存结果 Excel...")
df.to_excel(output_path, index=False)
print(f"\n✅ 全部完成！包含标准 [-1, 1] 归一化曲线的结果已保存至：{output_path}")