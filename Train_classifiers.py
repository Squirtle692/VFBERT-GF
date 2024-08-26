import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pickle
import torch.nn.functional as F
from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter


# Custom classes for models
class MultiKernelConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels_per_kernel, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels_per_kernel, kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)


class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = MultiKernelConvLayer(1, 32, [3, 5, 7])
        self.conv2 = MultiKernelConvLayer(32 * 3, 64, [3, 5, 7])
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64 * 3 * input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim, label_size)

    def forward(self, embeddings):
        lstm_out, _ = self.lstm(embeddings)
        label_space = self.hidden2label(lstm_out[:, -1, :])
        return label_space


# Functions to get models based on type
def get_model(model_type, input_dim=None, num_classes=3):
    if model_type == 'CNN':
        return CNNClassifier(input_dim, num_classes)
    elif model_type == 'LSTM':
        return LSTMClassifier(embedding_dim=input_dim, hidden_dim=50, label_size=num_classes)
    elif model_type == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(768, 256, 64), activation='relu', solver='adam', max_iter=500,
                             random_state=42)
    elif model_type == 'RF':
        return RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_leaf=4, class_weight='balanced',
                                      random_state=42)
    elif model_type == 'SGD':
        return SGDClassifier(random_state=42, loss='log')
    elif model_type == 'XGB':
        return XGBClassifier(max_depth=14, learning_rate=0.05, objective='multi:softprob', n_estimators=500,
                             subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', use_label_encoder=False,
                             num_class=num_classes, random_state=42)


# Training and evaluation function
def train_and_evaluate_model(model, train_features, train_labels, val_features, val_labels, model_type, fold):
    if model_type in ['MLP', 'RF', 'SGD', 'XGB']:
        model.fit(train_features, train_labels)
        val_probs = model.predict_proba(val_features)
        val_preds = np.argmax(val_probs, axis=1)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_features = torch.tensor(train_features).float().to(device)
        val_features = torch.tensor(val_features).float().to(device)
        train_labels = torch.tensor(train_labels).long().to(device)
        val_labels = torch.tensor(val_labels).long().to(device)

        train_dataset = TensorDataset(train_features, train_labels)
        val_dataset = TensorDataset(val_features, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        for epoch in range(30):
            model.train()
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        val_probs = []
        with torch.no_grad():
            for features, _ in val_loader:
                outputs = model(features)
                val_probs.append(outputs.cpu().numpy())

        val_probs = np.vstack(val_probs)
        val_preds = np.argmax(val_probs, axis=1)

    val_accuracy = accuracy_score(val_labels.cpu().numpy(), val_preds)
    val_f1 = f1_score(val_labels.cpu().numpy(), val_preds, average='macro')
    val_mcc = matthews_corrcoef(val_labels.cpu().numpy(), val_preds)

    print(
        f"Model: {model_type} | Fold: {fold + 1} | Validation Accuracy: {val_accuracy:.4f} | F1 Score: {val_f1:.4f} | MCC: {val_mcc:.4f}")

    return model, train_probs, val_probs


# Main execution logic
if __name__ == '__main__':
    device = torch.device("cuda:3")
    NUM_CLASSES = 3
    MODEL_TYPES = ['CNN', 'LSTM', 'MLP', 'RF', 'SGD', 'XGB']
    OUTPUT_DIR = "Probabilities"

    with open('VFbert_VFDB_ALL.pkl', 'rb') as f:
        bert_features = pickle.load(f)
    df = pd.read_csv('Combined_VFDB_data.tsv', delimiter='\t', header=0)
    labels_np = np.array(df.iloc[:, 2].tolist())
    bert_features_np = np.array(list(bert_features.values()))

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = [(train_index, val_index) for train_index, val_index in kfold.split(bert_features_np, labels_np)]

    for fold, (train_index, val_index) in enumerate(fold_splits):
        train_labels_fold = labels_np[train_index]
        val_labels_fold = labels_np[val_index]

        print(f"Fold {fold + 1}:")
        print("  Train label distribution:", Counter(train_labels_fold))
        print("  Validation label distribution:", Counter(val_labels_fold))

        for model_type in MODEL_TYPES:
            print(f"Training {model_type} - Fold {fold + 1}")

            if model_type in ['CNN', 'LSTM']:
                train_features = torch.from_numpy(bert_features_np[train_index]).float().unsqueeze(1)
                val_features = torch.from_numpy(bert_features_np[val_index]).float().unsqueeze(1)
            else:
                train_features = bert_features_np[train_index]
                val_features = bert_features_np[val_index]

            train_labels, val_labels = labels_np[train_index], labels_np[val_index]
            model = get_model(model_type, input_dim=train_features.shape[-1], num_classes=NUM_CLASSES)
            if model_type in ['CNN', 'LSTM']:
                model = model.to(device)
                model, train_probs, val_probs = train_and_evaluate_model(model, train_features, train_labels,
                                                                         val_features, val_labels, model_type, fold)
            else:
                model, train_probs, val_probs = train_and_evaluate_model(model, train_features, train_labels,
                                                                         val_features, val_labels, model_type, fold)

            # Save probabilities along with labels
            np.save(os.path.join(OUTPUT_DIR, f"{model_type}_train_probs_fold_{fold + 1}.npy"),
                    np.column_stack((train_probs, train_labels)))
            np.save(os.path.join(OUTPUT_DIR, f"{model_type}_val_probs_fold_{fold + 1}.npy"),
                    np.column_stack((val_probs, val_labels)))

            print(f"Train probabilities shape for {model_type} - Fold {fold + 1}: {train_probs.shape}")
            print(f"Validation probabilities shape for {model_type} - Fold {fold + 1}: {val_probs.shape}")
