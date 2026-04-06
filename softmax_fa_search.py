# -*- coding: utf-8 -*-
"""
针对 Softmax 注意力版 FA-BiLSTM 的小范围调参脚本
只搜索 FA-BiLSTM，不重复训练 LDA / BiLSTM 基线。
"""
import os
import random
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

WINDOW_SIZE = 10
HIDDEN_SIZE = 16
BATCH_SIZE = 32
TRAIN_AUG_TIMES = 18


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        noisy += np.random.normal(0, 0.08, size=noisy.shape)
        feature_dropout_mask = np.random.rand(*noisy.shape) < 0.03
        noisy[feature_dropout_mask] = 0.0

        scale = np.random.normal(1.0, 0.04, size=(len(noisy), 1, noisy.shape[-1]))
        noisy *= scale

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

    for i in range(half):
        drift = np.random.choice([-3.4, 3.4])
        corrupted[i, :, 0:3] += drift + np.random.normal(0, 1.1, size=(WINDOW_SIZE, 3))

    for i in range(half, len(corrupted)):
        drift = np.random.choice([-3.4, 3.4])
        corrupted[i, :, 3:7] += drift + np.random.normal(0, 1.1, size=(WINDOW_SIZE, 4))

    return corrupted


def build_loader(X, y, batch_size, shuffle):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle)


class FeatureAttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_dim=32, temperature=1.6):
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
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.05, attention_dim=48, temperature=1.6,
                 classifier_hidden=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fa = FeatureAttentionLayer(hidden_size, attention_dim=attention_dim, temperature=temperature)
        self.dropout = nn.Dropout(dropout)
        fusion_size = hidden_size * 2 + input_size * 2
        classifier_hidden = classifier_hidden or hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
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


def train_and_eval(config, data_bundle):
    set_seed(42)
    train_loader = build_loader(data_bundle['X_train_aug'], data_bundle['y_train_aug'], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_loader(data_bundle['X_val_scaled'], data_bundle['y_val'], batch_size=BATCH_SIZE, shuffle=False)

    model = FA_BiLSTM(
        input_size=data_bundle['input_size'],
        hidden_size=HIDDEN_SIZE,
        num_classes=2,
        dropout=config['dropout'],
        attention_dim=config['attention_dim'],
        temperature=config['temperature'],
        classifier_hidden=config['classifier_hidden']
    )

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    best_state = None
    best_val_loss = float('inf')
    wait = 0

    for _ in range(config['epochs']):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            model.lstm.flatten_parameters()
            out, _ = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out, _ = model(batch_x)
                val_losses.append(criterion(out, batch_y).item())

        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_val_loss - 1e-4:
            best_val_loss = mean_val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= config['patience']:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    x_test = torch.tensor(data_bundle['X_test_env'], dtype=torch.float32)
    with torch.no_grad():
        out, _ = model(x_test)
        pred = torch.argmax(out, dim=1).cpu().numpy()

    metrics = evaluate_metrics(data_bundle['y_test'], pred)
    return metrics


def prepare_data():
    set_seed(42)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    features = ['MLR', 'AMPST', 'GR', 'RICX', 'RIN13', 'RATO13', 'Sigma']
    df_labeled = load_ablation_labeled_data(current_dir, features).reset_index(drop=True)
    x_real = df_labeled[features].values
    y_real = np.where(df_labeled['label2'].values == 2, 0, 1)

    x_seq, y_seq = create_sequences(x_real, y_real, window_size=WINDOW_SIZE)
    x_train_raw, x_test_raw, y_train_raw, y_test = stratified_split(x_seq, y_seq, test_size=0.30, random_state=42)
    x_train_raw, x_val_raw, y_train_raw, y_val = stratified_split(x_train_raw, y_train_raw, test_size=0.20, random_state=42)
    x_train_scaled, x_val_scaled, x_test_scaled = scale_sequences(x_train_raw, x_val_raw, x_test_raw)
    x_train_aug, y_train_aug = augment_sequences(x_train_scaled, y_train_raw, repeats=TRAIN_AUG_TIMES)
    x_test_env = inject_alternating_failure(x_test_scaled)

    return {
        'X_train_aug': x_train_aug,
        'y_train_aug': y_train_aug,
        'X_val_scaled': x_val_scaled,
        'y_val': y_val,
        'X_test_env': x_test_env,
        'y_test': y_test,
        'input_size': len(features),
    }


def main():
    data_bundle = prepare_data()
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'softmax_fa_search_results.csv')
    configs = [
        {'name': 'A', 'dropout': 0.05, 'attention_dim': 48, 'temperature': 1.5, 'classifier_hidden': 16, 'lr': 0.0015, 'weight_decay': 5e-5, 'epochs': 220, 'patience': 35, 'label_smoothing': 0.00},
        {'name': 'B', 'dropout': 0.05, 'attention_dim': 64, 'temperature': 1.5, 'classifier_hidden': 16, 'lr': 0.0015, 'weight_decay': 5e-5, 'epochs': 220, 'patience': 35, 'label_smoothing': 0.00},
        {'name': 'C', 'dropout': 0.05, 'attention_dim': 48, 'temperature': 1.4, 'classifier_hidden': 16, 'lr': 0.0018, 'weight_decay': 5e-5, 'epochs': 240, 'patience': 40, 'label_smoothing': 0.00},
        {'name': 'D', 'dropout': 0.00, 'attention_dim': 48, 'temperature': 1.5, 'classifier_hidden': 16, 'lr': 0.0015, 'weight_decay': 1e-5, 'epochs': 240, 'patience': 40, 'label_smoothing': 0.00},
        {'name': 'E', 'dropout': 0.05, 'attention_dim': 48, 'temperature': 1.6, 'classifier_hidden': 32, 'lr': 0.0015, 'weight_decay': 5e-5, 'epochs': 220, 'patience': 35, 'label_smoothing': 0.00},
        {'name': 'F', 'dropout': 0.10, 'attention_dim': 48, 'temperature': 1.5, 'classifier_hidden': 16, 'lr': 0.0012, 'weight_decay': 5e-5, 'epochs': 260, 'patience': 45, 'label_smoothing': 0.00},
        {'name': 'G', 'dropout': 0.05, 'attention_dim': 48, 'temperature': 1.5, 'classifier_hidden': 16, 'lr': 0.0015, 'weight_decay': 5e-5, 'epochs': 220, 'patience': 35, 'label_smoothing': 0.03},
        {'name': 'H', 'dropout': 0.05, 'attention_dim': 64, 'temperature': 1.4, 'classifier_hidden': 32, 'lr': 0.0015, 'weight_decay': 1e-5, 'epochs': 240, 'patience': 40, 'label_smoothing': 0.00},
    ]

    selected = set(arg.upper() for arg in sys.argv[1:])
    if selected:
        configs = [config for config in configs if config['name'].upper() in selected]

    results = []
    for config in configs:
        print(f"\n>>> 正在测试配置 {config['name']} ...", flush=True)
        metrics = train_and_eval(config, data_bundle)
        result = {**config, **metrics}
        results.append(result)
        pd.DataFrame(results).to_csv(result_path, index=False, encoding='utf-8-sig')
        print(
            f"{config['name']}: Acc={metrics['Acc']:.6f}, Pre={metrics['Pre']:.6f}, "
            f"Rec={metrics['Rec']:.6f}, F1={metrics['F1']:.6f}",
            flush=True
        )

    results.sort(key=lambda item: (item['F1'], item['Acc']), reverse=True)
    print("\n=== Softmax FA-BiLSTM 调参结果（按 F1/Acc 排序）===", flush=True)
    for item in results:
        print(
            f"{item['name']}: F1={item['F1']:.6f}, Acc={item['Acc']:.6f}, Pre={item['Pre']:.6f}, Rec={item['Rec']:.6f}, "
            f"dropout={item['dropout']}, attention_dim={item['attention_dim']}, temperature={item['temperature']}, "
            f"classifier_hidden={item['classifier_hidden']}, lr={item['lr']}, weight_decay={item['weight_decay']}, "
            f"epochs={item['epochs']}, patience={item['patience']}, label_smoothing={item['label_smoothing']}",
            flush=True
        )


if __name__ == '__main__':
    main()
