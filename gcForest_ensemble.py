import numpy as np
import os
from sklearn.metrics import accuracy_score
from joblib import dump, load
from deepforest import CascadeForestClassifier
from sklearn.model_selection import StratifiedKFold

NUM_FOLDS = 5
MODEL_TYPES = ['CNN', 'LSTM', 'MLP', 'RF', 'SGD', 'XGB']
OUTPUT_DIR = "Probabilities_VFDB"
GCFOREST_MODEL_DIR = "Trained_Model_VFDB"


def load_probabilities(model_type, fold, dataset_type="val"):
    """Load saved val probabilities from disk."""
    file_path = f'{OUTPUT_DIR}/{model_type}_{dataset_type}_probs_fold_{fold}.npy'
    return np.load(file_path)


def train_gcforest_on_fold(fold, X_train, y_train):
    """Train a gcForest model on the provided training data."""
    y_train = y_train.astype(int)
    gc_model = CascadeForestClassifier(predictor="forest", random_state=5)
    gc_model.fit(X_train, y_train)

    # Save the trained model
    model_path = os.path.join(GCFOREST_MODEL_DIR, f'CascadeForest_fold_{fold}.joblib')
    if not os.path.exists(GCFOREST_MODEL_DIR):
        os.makedirs(GCFOREST_MODEL_DIR) 
    dump(gc_model, model_path)
    print(f"Model for fold {fold} saved to {model_path}")

    return gc_model


def evaluate_gcforest(model, X, y):
    """Evaluate the gcForest model and return the accuracy."""
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)


def combine_probabilities(prob_list):
    """Combine the probabilities from different models into a single feature set."""
    features = np.hstack([prob[:, :-1] for prob in prob_list])
    labels = prob_list[0][:, -1].astype(int)
    return features, labels


def combine_all_folds_probabilities():
    """Combine all val probabilities from all folds to form a full dataset."""
    combined_features, combined_labels = [], []

    for fold in range(1, NUM_FOLDS + 1):
        val_prob_list = [load_probabilities(model_type, fold, "val") for model_type in MODEL_TYPES]
        X_val, y_val = combine_probabilities(val_prob_list)
        combined_features.append(X_val)
        combined_labels.append(y_val)

    combined_features = np.vstack(combined_features)
    combined_labels = np.hstack(combined_labels)

    return combined_features, combined_labels


def main():
    # Combine all fold val datasets to form a new full dataset
    combined_features, combined_labels = combine_all_folds_probabilities()

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_splits = skf.split(combined_features, combined_labels)

    train_accs, val_accs = [], []

    for fold, (train_idx, val_idx) in enumerate(fold_splits, 1):
        print(f"Processing Fold {fold}")

        X_train, y_train = combined_features[train_idx], combined_labels[train_idx]
        X_val, y_val = combined_features[val_idx], combined_labels[val_idx]

        gc_model = train_gcforest_on_fold(fold, X_train, y_train)
        train_acc = evaluate_gcforest(gc_model, X_train, y_train)
        val_acc = evaluate_gcforest(gc_model, X_val, y_val)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Fold {fold} - Training Accuracy: {train_acc:.4f}')
        print(f'Fold {fold} - Validation Accuracy: {val_acc:.4f}')

    print(f'Average Training Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}')
    print(f'Average Validation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}')


if __name__ == "__main__":
    main()
