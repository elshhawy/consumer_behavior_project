import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, preprocess_data, FIGURES_PATH, REPORTS_PATH

# ── Build Deep Learning Model ───────────────────────────
def build_dl_model(input_dim):
    inputs = Input(shape=(input_dim,))

    # Layer 1
    x = Dense(512, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Layer 2
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Layer 3
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Layer 4
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Layer 5 - Feature Extraction Layer
    features = Dense(32, activation='relu', name='feature_layer')(x)

    # Output
    output = Dense(1, activation='sigmoid', name='output')(features)

    model = Model(inputs=inputs, outputs=output)
    return model

# ── Extract Deep Features ───────────────────────────────
def extract_features(model, X):
    feature_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer('feature_layer').output
    )
    return feature_extractor.predict(X, verbose=0)

# ── Plot Training History ───────────────────────────────
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', color='blue')
    axes[0].plot(history.history['val_accuracy'], label='Validation', color='red')
    axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', color='blue')
    axes[1].plot(history.history['val_loss'], label='Validation', color='red')
    axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/deep_learning_training.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print(" Training History saved!")

# ── Plot Confusion Matrix ───────────────────────────────
def plot_confusion_matrix_dl(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['No Purchase', 'Purchase'],
                yticklabels=['No Purchase', 'Purchase'])
    plt.title('Confusion Matrix - Deep Learning', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/confusion_matrix_deep_learning.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print(" DL Confusion Matrix saved!")

# ── Train Deep Learning Model ───────────────────────────
def train_deep_learning(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print(" Training Deep Learning Model...")
    print("="*50)

    # Convert to numpy
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    # Build model
    model = build_dl_model(X_train_np.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, verbose=1)
    ]

    history = model.fit(
        X_train_np, y_train_np,
        epochs=300,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred_prob = model.predict(X_test_np, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    report = classification_report(y_test_np, y_pred,
                                   target_names=['No Purchase', 'Purchase'])
    cm = confusion_matrix(y_test_np, y_pred)
    roc_auc = roc_auc_score(y_test_np, y_pred_prob)
    accuracy = (y_pred == y_test_np).mean()

    print(f"\n Classification Report - Deep Learning:")
    print(report)
    print(f" ROC AUC: {roc_auc:.3f}")
    print(f" Accuracy: {accuracy:.3f}")

    # Save report
    with open(f"{REPORTS_PATH}/classification_report_deep_learning.txt", 'w') as f:
        f.write("Classification Report - Deep Learning\n")
        f.write("="*50 + "\n")
        f.write(report)
        f.write(f"\nROC AUC: {roc_auc:.3f}")
        f.write(f"\nAccuracy: {accuracy:.3f}")

    # Plots
    plot_training_history(history)
    plot_confusion_matrix_dl(cm)

    # Extract deep features for next phase
    deep_features_train = extract_features(model, X_train_np)
    deep_features_test = extract_features(model, X_test_np)

    print(f"\n Deep Features Shape (Train): {deep_features_train.shape}")
    print(f" Deep Features Shape (Test):  {deep_features_test.shape}")

    # Save features
    np.save(f"{REPORTS_PATH}/deep_features_train.npy", deep_features_train)
    np.save(f"{REPORTS_PATH}/deep_features_test.npy", deep_features_test)
    np.save(f"{REPORTS_PATH}/y_train.npy", y_train_np)
    np.save(f"{REPORTS_PATH}/y_test.npy", y_test_np)
    print(" Deep Features saved for next phase!")

    return model, deep_features_train, deep_features_test, roc_auc, accuracy

# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)
    model, train_features, test_features, roc_auc, acc = train_deep_learning(
        X_train, X_test, y_train, y_test
    )