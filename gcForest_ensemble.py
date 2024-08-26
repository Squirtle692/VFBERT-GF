import numpy as np
import os
from sklearn.metrics import accuracy_score
from joblib import dump, load
from deepforest import CascadeForestClassifier

NUM_FOLDS = 5
MODEL_TYPES = ['CNN', 'LSTM', 'MLP', 'RF', 'SGD', 'XGB']
OUTPUT_DIR = "Probabilities"
GCFOREST_MODEL_DIR = "Trained_Model"

def load_probabilities(model_type, fold, dataset_type="train"):
    """Load saved probabilities from disk."""
    file_path = f'{OUTPUT_DIR}/{model_type}_{dataset_type}_probs_fold_{fold}.npy'
    return np.load(file_path)

def train_gcforest_on_fold(fold, X_train, y_train):
    """Train a gcForest model on the provided training data."""
    y_train = y_train.astype(int)
    gc_model = CascadeForestClassifier(predictor="forest", random_state=42)
    gc_model.fit(X_train, y_train)
    model_path = os.path.join(GCFOREST_MODEL_DIR, f'CascadeForest_fold_{fold}.joblib')
    dump(gc_model, model_path)
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

def main():
    train_accs, val_accs = [], []

    for fold in range(1, NUM_FOLDS + 1):
        print(f"Processing Fold {fold}")

        train_prob_list = [load_probabilities(model_type, fold, "train") for model_type in MODEL_TYPES]
        val_prob_list = [load_probabilities(model_type, fold, "val") for model_type in MODEL_TYPES]

        X_train, y_train = combine_probabilities(train_prob_list)
        X_val, y_val = combine_probabilities(val_prob_list)

        gc_model = train_gcforest_on_fold(fold, X_train, y_train)
        train_acc = evaluate_gcforest(gc_model, X_train, y_train)
        val_acc = evaluate_gcforest(gc_model, X_val, y_val)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Fold {fold} - Training Accuracy: {train_acc:.3f}')
        print(f'Fold {fold} - Validation Accuracy: {val_acc:.3f}')

    print(f'Average Training Accuracy: {np.mean(train_accs):.3f} ± {np.std(train_accs):.3f}')
    print(f'Average Validation Accuracy: {np.mean(val_accs):.3f} ± {np.std(val_accs):.3f}')

    test_accuracies = []

    for fold in range(1, NUM_FOLDS + 1):
        test_prob_list = [load_probabilities(model_type, fold, "val") for model_type in MODEL_TYPES]
        X_test, y_test = combine_probabilities(test_prob_list)
        gc_model = load(os.path.join(GCFOREST_MODEL_DIR, f'CascadeForest_fold_{fold}.joblib'))
        test_accuracy = evaluate_gcforest(gc_model, X_test, y_test)
        test_accuracies.append(test_accuracy)
        print(f'Fold {fold} Test Accuracy: {test_accuracy:.3f}')

    print(f'Average Test Accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}')

if __name__ == "__main__":
    main()
