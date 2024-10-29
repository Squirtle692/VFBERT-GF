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
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter

MODEL_TYPES = ['CNN', 'LSTM', 'MLP', 'RF', 'SGD', 'XGB']
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

def get_model(model_type, input_dim=None, num_classes=3):
    if model_type == 'CNN':
        return CNNClassifier(input_dim, num_classes)
    elif model_type == 'LSTM':
        return LSTMClassifier(embedding_dim=input_dim, hidden_dim=50, label_size=num_classes)
    elif model_type == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(768, 512, 128), activation='relu', solver='adam', max_iter=100,
                             random_state=42)
    elif model_type == 'RF':
        return RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_leaf=4, class_weight='balanced',
                                      random_state=42)
    elif model_type == 'SGD':
        return SGDClassifier(random_state=42, loss='log_loss')
    elif model_type == 'XGB':
        return XGBClassifier(max_depth=6, learning_rate=0.05, objective='multi:softprob', n_estimators=100,
                             subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', use_label_encoder=False,
                             num_class=num_classes, random_state=42)

def train_and_evaluate_model(model, train_features, train_labels, val_features, val_labels, model_type, fold):
    # Placeholder for probabilities
    train_probs = []
    val_probs = []

    if model_type in ['MLP', 'RF', 'SGD', 'XGB']:
        model.fit(train_features, train_labels)
        train_probs = model.predict_proba(train_features)
        val_probs = model.predict_proba(val_features)
        val_preds = np.argmax(val_probs, axis=1)
        val_labels_np = val_labels
    else:
        train_features = train_features.float()
        train_labels = torch.tensor(train_labels).long()
        val_features = val_features.float()
        val_labels = torch.tensor(val_labels).long()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_dataset = TensorDataset(train_features, train_labels)
        val_dataset = TensorDataset(val_features, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        model = model.to(device)

        for epoch in range(30):
            model.train()
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for features, labels in train_loader:
                features = features.to(device)
                outputs = model(features)
                train_probs.append(outputs.cpu().numpy())

            for features, labels in val_loader:
                features = features.to(device)
                outputs = model(features)
                val_probs.append(outputs.cpu().numpy())

        train_probs = np.vstack(train_probs)
        val_probs = np.vstack(val_probs)

        val_preds = np.argmax(val_probs, axis=1)
        val_labels_np = val_labels.cpu().numpy() if isinstance(val_labels, torch.Tensor) else val_labels

    val_accuracy = accuracy_score(val_labels_np, val_preds)
    val_f1 = f1_score(val_labels_np, val_preds, average='macro')
    val_mcc = matthews_corrcoef(val_labels_np, val_preds)

    print(f"Model: {model_type} | Fold: {fold + 1} | Validation Accuracy: {val_accuracy:.4f} | F1 Score: {val_f1:.4f} | MCC: {val_mcc:.4f}")

    return model, train_probs, val_probs, val_accuracy

model_accuracies_victors = {model_type: [] for model_type in MODEL_TYPES}
model_accuracies = {model_type: [] for model_type in MODEL_TYPES}

if __name__ == '__main__':
    device = torch.device("cuda:0")
    NUM_CLASSES = 3
    OUTPUT_DIR = "VFDB_Probabilities"

    with open('data/VFbert_VFDB_ALL.pkl', 'rb') as f:
        bert_features = pickle.load(f)
    df = pd.read_csv('data/Combined_VFDB_data.tsv', delimiter='\t', header=0)
    labels_np = np.array(df.iloc[:, 2].tolist())
    bert_features_np = np.array(list(bert_features.values()))

    with open('data/VFbert_Victors_ALL.pkl', 'rb') as f_victors:
        victors_features = pickle.load(f_victors)
    df_victors = pd.read_csv('data/Combined_Victors_data.tsv', delimiter='\t', header=0)
    victors_labels_np = np.array(df_victors.iloc[:, 1].tolist())
    victors_features_np = np.array(list(victors_features.values()))

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
                train_labels, val_labels = labels_np[train_index], labels_np[val_index]

                model = get_model(model_type, input_dim=train_features.shape[-1], num_classes=NUM_CLASSES)
                model = model.to(device)
                model, train_probs, val_probs, val_accuracy = train_and_evaluate_model(
                    model, train_features, train_labels, val_features, val_labels, model_type, fold)
            else:
                train_features = bert_features_np[train_index]
                val_features = bert_features_np[val_index]
                train_labels, val_labels = labels_np[train_index], labels_np[val_index]

                model = get_model(model_type, input_dim=train_features.shape[-1], num_classes=NUM_CLASSES)
                model, train_probs, val_probs, val_accuracy = train_and_evaluate_model(
                    model, train_features, train_labels, val_features, val_labels, model_type, fold)

            model_accuracies[model_type].append(val_accuracy)

            if model_type in ['CNN', 'LSTM']:
                victors_features_fold = torch.from_numpy(victors_features_np).float().unsqueeze(1).to(device)
                victors_probs = model(victors_features_fold).detach().cpu().numpy()
            else:
                victors_features_fold = victors_features_np
                victors_probs = model.predict_proba(victors_features_fold)

            victors_preds = np.argmax(victors_probs, axis=1)
            victors_accuracy = accuracy_score(victors_labels_np, victors_preds)
            model_accuracies_victors[model_type].append(victors_accuracy)

            print(f"Model: {model_type} | Fold: {fold + 1} | Victors Accuracy: {victors_accuracy:.4f}")

            np.save(os.path.join(OUTPUT_DIR, f"{model_type}_val_probs_fold_{fold + 1}.npy"), np.column_stack((val_probs, val_labels)))
            np.save(os.path.join(OUTPUT_DIR, f"{model_type}_victors_probs_fold_{fold + 1}.npy"), np.column_stack((victors_probs, victors_labels_np)))


    for model_type in MODEL_TYPES:
        average_accuracy = np.mean(model_accuracies[model_type])
        accuracy_std_dev = np.std(model_accuracies[model_type])
        print(f"Average Accuracy for {model_type} on BVBRC: {average_accuracy:.4f} ± {accuracy_std_dev:.4f}")

    for model_type in MODEL_TYPES:
        average_accuracy_victors = np.mean(model_accuracies_victors[model_type])
        accuracy_std_dev_victors = np.std(model_accuracies_victors[model_type])
        print(f"Average Accuracy for {model_type} on Victors: {average_accuracy_victors:.4f} ± {accuracy_std_dev_victors:.4f}")
