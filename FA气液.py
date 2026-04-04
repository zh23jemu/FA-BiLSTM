# -*- coding: utf-8 -*-
"""
渤海深层潜山全井段流体智能评价系统
核心：基于 FA-BiLSTM 输出连续流体指示曲线与动态特征权重
"""
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 0. 全局设置与超参数
# ==========================================
WINDOW_SIZE = 10
HIDDEN_SIZE = 16
ATTENTION_DIM = 32
BATCH_SIZE = 32
EPOCHS = 180
PATIENCE = 25
LR = 0.003
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
NOISE_STD = 0.05


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_sequences(X, labels, window_size):
    X_seq_all = []
    valid_seq_indices = []
    y_train_list = []

    for i in range(len(X) - window_size + 1):
        X_seq_all.append(X[i: i + window_size])
        target_idx = i + window_size - 1
        label = labels.iloc[target_idx]
        if pd.notna(label) and label in [2, 3]:
            valid_seq_indices.append(i)
            y_train_list.append(0 if label == 2 else 1)

    return np.array(X_seq_all), np.array(valid_seq_indices), np.array(y_train_list)


def build_loader(X, y, batch_size, shuffle):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle)


class FeatureAttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_dim=32):
        super().__init__()
        self.W_h = nn.Linear(hidden_size * 2, attention_dim, bias=False)
        self.W_x = nn.Linear(1, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=True)

    def forward(self, h_context, x_current):
        h_proj = self.W_h(h_context).unsqueeze(1)
        x_proj = self.W_x(x_current.unsqueeze(-1))
        energy = torch.tanh(h_proj + x_proj)
        score = self.v(energy).squeeze(-1)
        alpha = torch.sigmoid(score)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        weighted_x = alpha * x_current
        return alpha, weighted_x


class FA_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, attention_dim=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fa = FeatureAttentionLayer(hidden_size, attention_dim)
        self.dropout = nn.Dropout(dropout)
        fusion_size = hidden_size * 2 + input_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h_target = out[:, -1, :]
        x_target = x[:, -1, :]
        alpha, weighted_x = self.fa(h_target, x_target)
        fused = torch.cat([self.dropout(h_target), x_target, weighted_x], dim=1)
        return self.classifier(fused), alpha


def train_model(model, train_loader, val_loader, class_weights):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_state = None
    best_val_loss = float('inf')
    wait = 0

    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            model.lstm.flatten_parameters()

            noisy_batch = batch_x + torch.randn_like(batch_x) * NOISE_STD
            feature_dropout = (torch.rand_like(noisy_batch) < 0.02)
            noisy_batch = noisy_batch.masked_fill(feature_dropout, 0.0)

            logits, _ = model(noisy_batch)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits, _ = model(batch_x)
                val_losses.append(criterion(logits, batch_y).item())

        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_val_loss - 1e-4:
            best_val_loss = mean_val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


set_seed(42)

# ==========================================
# 1. 加载数据并进行全井段预处理
# ==========================================
print(">>> [1/5] 正在加载全井段测井数据...")
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '6-3训练数据_副本3.xlsx')
output_path = os.path.join(current_dir, 'FA_BiLSTM_连续波动曲线_全表.xlsx')

df = pd.read_excel(file_path)
features = ['Sigma', 'RICX', 'RIN13', 'RATO13']

df[features] = df[features].ffill().bfill()
X_all = df[features].values

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# ==========================================
# 2. 构建深度序列滑动窗口
# ==========================================
print(">>> [2/5] 正在构建时序深度滑动窗口...")
X_seq_all, valid_seq_indices, y_train_seq = create_sequences(X_all_scaled, df['label2'], WINDOW_SIZE)
X_train_seq = X_seq_all[valid_seq_indices]

class_counts = np.bincount(y_train_seq)
weights = len(y_train_seq) / (len(class_counts) * class_counts)
class_weights = torch.tensor(weights, dtype=torch.float32)

train_idx, val_idx = train_test_split(
    np.arange(len(y_train_seq)),
    test_size=0.2,
    random_state=42,
    stratify=y_train_seq
)

train_loader = build_loader(X_train_seq[train_idx], y_train_seq[train_idx], BATCH_SIZE, True)
val_loader = build_loader(X_train_seq[val_idx], y_train_seq[val_idx], BATCH_SIZE, False)

# ==========================================
# 3. 定义并训练网络
# ==========================================
print(f">>> [3/5] 正在基于 {len(y_train_seq)} 个已知标签段进行网络训练...")
model = FA_BiLSTM(
    input_size=len(features),
    hidden_size=HIDDEN_SIZE,
    num_classes=2,
    attention_dim=ATTENTION_DIM,
    dropout=DROPOUT
)
model = train_model(model, train_loader, val_loader, class_weights)

# ==========================================
# 4. 全井段连续推演
# ==========================================
print(">>> [4/5] 正在面向整口井进行连续波动推演...")
model.eval()
X_all_t = torch.tensor(X_seq_all, dtype=torch.float32)

with torch.no_grad():
    logits, alphas = model(X_all_t)
    gas_logits = logits[:, 0].numpy()

    scaler_gas = MinMaxScaler(feature_range=(-1, 1))
    gas_logits_scaled = scaler_gas.fit_transform(gas_logits.reshape(-1, 1)).flatten()

    temperature = 3.0
    probs_smooth = F.softmax(logits / temperature, dim=1).numpy()
    preds = np.argmax(F.softmax(logits, dim=1).numpy(), axis=1).astype(float)
    alphas_np = alphas.numpy()

pad_len = WINDOW_SIZE - 1
pred_curve = np.pad(preds, (pad_len, 0), constant_values=np.nan)
pred_label_mapped = np.where(pred_curve == 0, 2, np.where(pred_curve == 1, 3, np.nan))

gas_index_curve = np.pad(gas_logits_scaled, (pad_len, 0), constant_values=np.nan)
df['FA_Gas_Index'] = gas_index_curve

if 'I_reservoir' in df.columns:
    df.loc[df['I_reservoir'] < 0, 'FA_Gas_Index'] = 0
else:
    print(">>> 警告: 原数据中未发现 'I_reservoir' 列，请检查表头名是否一致！")

df['FA_p_gas_smooth'] = np.pad(probs_smooth[:, 0], (pad_len, 0), constant_values=np.nan)
df['FA_pred_label2'] = pred_label_mapped

for i, feat in enumerate(features):
    alpha_curve = np.pad(alphas_np[:, i], (pad_len, 0), constant_values=np.nan)
    df[f'FA_alpha_{feat}'] = alpha_curve

# ==========================================
# 5. 导出结果
# ==========================================
print(">>> [5/5] 正在保存结果 Excel...")
df.to_excel(output_path, index=False)
print(f"全部完成，结果已保存至：{output_path}")
