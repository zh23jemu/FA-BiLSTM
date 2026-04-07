# -*- coding: utf-8 -*-
"""
渤海深层潜山多参数流体智能评价系统
基于真实时序窗口、训练集增强与交替仪器失效干扰的消融实验
"""
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# ==========================================
# 0. 全局设置
# ==========================================
WINDOW_SIZE = 10
HIDDEN_SIZE = 16
BATCH_SIZE = 32
EPOCHS = 180
LR = 0.003
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
PATIENCE = 25
TRAIN_AUG_TIMES = 18


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


def create_sequences(X, y, window_size=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i: i + window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)


def stratified_split(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_sequences(train_seq, val_seq, test_seq):
    scaler = StandardScaler()
    scaler.fit(train_seq.reshape(-1, train_seq.shape[-1]))

    def transform(x):
        return scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

    return transform(train_seq), transform(val_seq), transform(test_seq)


def augment_sequences(X, y, repeats=12):
    augmented_x = [X]
    augmented_y = [y]

    for _ in range(repeats):
        noisy = X.copy()

        # 常规噪声与轻微通道失活
        noisy += np.random.normal(0, 0.08, size=noisy.shape)
        feature_dropout_mask = np.random.rand(*noisy.shape) < 0.03
        noisy[feature_dropout_mask] = 0.0

        # 随机尺度扰动，模拟仪器轻微漂移
        scale = np.random.normal(1.0, 0.04, size=(len(noisy), 1, noisy.shape[-1]))
        noisy *= scale

        # 分组基线漂移，模拟交替仪器异常
        group1_mask = np.random.rand(len(noisy)) < 0.22
        if group1_mask.any():
            drift = np.random.choice([-1.8, 1.8], size=group1_mask.sum()).reshape(-1, 1, 1)
            noisy[group1_mask, :, 0:3] += drift + np.random.normal(0, 0.45, size=(group1_mask.sum(), WINDOW_SIZE, 3))

        group2_mask = np.random.rand(len(noisy)) < 0.22
        if group2_mask.any():
            drift = np.random.choice([-1.8, 1.8], size=group2_mask.sum()).reshape(-1, 1, 1)
            noisy[group2_mask, :, 3:7] += drift + np.random.normal(0, 0.45, size=(group2_mask.sum(), WINDOW_SIZE, 4))

        augmented_x.append(noisy)
        augmented_y.append(y.copy())

    return np.concatenate(augmented_x, axis=0), np.concatenate(augmented_y, axis=0)


def inject_alternating_failure(X):
    corrupted = X.copy()
    half = len(corrupted) // 2

    # 前半段：常规测井仪漂移
    for i in range(half):
        drift = np.random.choice([-3.4, 3.4])
        corrupted[i, :, 0:3] += drift + np.random.normal(0, 1.1, size=(WINDOW_SIZE, 3))

    # 后半段：核物理测井仪漂移
    for i in range(half, len(corrupted)):
        drift = np.random.choice([-3.4, 3.4])
        corrupted[i, :, 3:7] += drift + np.random.normal(0, 1.1, size=(WINDOW_SIZE, 4))

    return corrupted


def build_loader(X, y, batch_size, shuffle):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class FeatureAttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_dim=32, temperature=1.5):
        super().__init__()
        self.W_h = nn.Linear(hidden_size * 2, attention_dim, bias=False)
        self.W_x = nn.Linear(1, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=True)
        self.temperature = temperature

    def forward(self, h_context, x_current):
        h_proj = self.W_h(h_context).unsqueeze(1)
        x_proj = self.W_x(x_current.unsqueeze(-1))
        energy = torch.tanh(h_proj + x_proj)
        score = self.v(energy).squeeze(-1)
        alpha = F.softmax(score / self.temperature, dim=1)
        weighted_x = alpha * x_current
        return alpha, weighted_x


class FA_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2, attention_dim=32, temperature=1.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fa = FeatureAttentionLayer(hidden_size, attention_dim=attention_dim, temperature=temperature)
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


def evaluate_metrics(y_true, pred):
    return {
        'Acc': accuracy_score(y_true, pred),
        'Pre': precision_score(y_true, pred, average='macro'),
        'Rec': recall_score(y_true, pred, average='macro'),
        'F1': f1_score(y_true, pred, average='macro')
    }


def evaluate_sklearn_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return evaluate_metrics(y_test, pred)


def load_ablation_labeled_data(current_dir, features):
    sources = []

    base_df = pd.read_excel(os.path.join(current_dir, '6-3训练数据_副本2.xlsx'))
    sources.append(base_df)

    extra_file = os.path.join(current_dir, '扩充数据.xlsx')
    if os.path.exists(extra_file):
        extra_df = pd.read_excel(extra_file, sheet_name='扩充数据2').copy()
        if 'SGFC' in extra_df.columns and 'Sigma' not in extra_df.columns:
            extra_df['Sigma'] = extra_df['SGFC']
        sources.append(extra_df)

    labeled_frames = []
    for df_source in sources:
        if set(features + ['label2']).issubset(df_source.columns):
            labeled_frames.append(df_source.dropna(subset=['label2']).copy())

    return pd.concat(labeled_frames, ignore_index=True)


def train_model(model, train_loader, val_loader, X_test_env, y_test, lr=LR, weight_decay=WEIGHT_DECAY,
                epochs=EPOCHS, patience=PATIENCE):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val_loss = float('inf')
    wait = 0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            model.lstm.flatten_parameters()

            out = model(batch_x)
            if isinstance(out, tuple):
                out = out[0]

            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out = model(batch_x)
                if isinstance(out, tuple):
                    out = out[0]
                val_losses.append(criterion(out, batch_y).item())

        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_val_loss - 1e-4:
            best_val_loss = mean_val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    X_test_env_t = torch.tensor(X_test_env, dtype=torch.float32)
    with torch.no_grad():
        out = model(X_test_env_t)
        if isinstance(out, tuple):
            out = out[0]
        pred = torch.argmax(out, dim=1).cpu().numpy()

    return evaluate_metrics(y_test, pred)


# ==========================================
# 1. 加载真实数据并构建时序样本
# ==========================================
print(">>> [1/4] 加载真实标注井段并构建时序窗口...")
current_dir = os.path.dirname(os.path.abspath(__file__))
features = ['MLR', 'AMPST', 'GR', 'RICX', 'RIN13', 'RATO13', 'Sigma']
df_labeled = load_ablation_labeled_data(current_dir, features).reset_index(drop=True)
X_real = df_labeled[features].values
y_real = np.where(df_labeled['label2'].values == 2, 0, 1)

X_seq, y_seq = create_sequences(X_real, y_real, window_size=WINDOW_SIZE)

X_train_raw, X_test_raw, y_train_raw, y_test = stratified_split(X_seq, y_seq, test_size=0.30, random_state=42)
X_train_raw, X_val_raw, y_train_raw, y_val = stratified_split(X_train_raw, y_train_raw, test_size=0.20, random_state=42)

X_train_scaled, X_val_scaled, X_test_scaled = scale_sequences(X_train_raw, X_val_raw, X_test_raw)
X_train_aug, y_train_aug = augment_sequences(X_train_scaled, y_train_raw, repeats=TRAIN_AUG_TIMES)

# ==========================================
# 2. 在测试集注入交替仪器失效
# ==========================================
print(">>> [2/4] 在保留时序结构的测试集上注入交替仪器失效...")
X_test_env = inject_alternating_failure(X_test_scaled)

# ==========================================
# 3. 传统机器学习基线评估
# ==========================================
print(">>> [3/4] 计算 LDA / Random Forest / XGBoost 在失效工况下的退化表现...")
X_train_tabular = X_train_scaled[:, -1, :]
X_test_tabular = X_test_env[:, -1, :]

lda = LinearDiscriminantAnalysis()
metrics_results = {
    'LDA': evaluate_sklearn_model(lda, X_train_tabular, y_train_raw, X_test_tabular, y_test)
}

rf = RandomForestClassifier(
    n_estimators=260,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
metrics_results['Random Forest'] = evaluate_sklearn_model(rf, X_train_tabular, y_train_raw, X_test_tabular, y_test)

xgb = XGBClassifier(
    n_estimators=220,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    min_child_weight=2,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=1
)
metrics_results['XGBoost'] = evaluate_sklearn_model(xgb, X_train_tabular, y_train_raw, X_test_tabular, y_test)

# ==========================================
# 4. 深度学习模型训练与评估
# ==========================================
print(">>> [4/4] 训练 BiLSTM / FA-BiLSTM 并进行鲁棒性评估...")
set_seed(42)
train_loader = build_loader(X_train_aug, y_train_aug, batch_size=BATCH_SIZE, shuffle=True)
val_loader = build_loader(X_val_scaled, y_val, batch_size=BATCH_SIZE, shuffle=False)
metrics_results['BiLSTM'] = train_model(
    BiLSTM(len(features), HIDDEN_SIZE, 2, dropout=0.35),
    train_loader,
    val_loader,
    X_test_env,
    y_test
)

set_seed(42)
train_loader = build_loader(X_train_aug, y_train_aug, batch_size=BATCH_SIZE, shuffle=True)
val_loader = build_loader(X_val_scaled, y_val, batch_size=BATCH_SIZE, shuffle=False)
metrics_results['FA-BiLSTM'] = train_model(
    FA_BiLSTM(len(features), HIDDEN_SIZE, 2, dropout=0.0, attention_dim=48, temperature=1.5),
    train_loader,
    val_loader,
    X_test_env,
    y_test,
    lr=0.0015,
    weight_decay=1e-5,
    epochs=240,
    patience=40
)

print("\n" + "=" * 72)
print(" 表 5-X 交替仪器失效工况下不同耦合模型的流体识别性能")
print("=" * 72)
print(f"{'模型名称':<12} | {'准确率(Acc)':<12} | {'精确率(Pre)':<12} | {'召回率(Rec)':<12} | {'F1分数(F1)':<12}")
print("-" * 72)
for name in ['LDA', 'Random Forest', 'XGBoost', 'BiLSTM', 'FA-BiLSTM']:
    res = metrics_results[name]
    print(f"{name:<14} | {res['Acc']:.3%}       | {res['Pre']:.3%}       | {res['Rec']:.3%}       | {res['F1']:.3%}")
print("=" * 72)
