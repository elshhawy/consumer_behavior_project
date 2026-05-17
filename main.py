import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_data, preprocess_data, plot_correlation_matrix
from models import get_models, train_and_evaluate
from deep_learning import train_deep_learning
from segmentation import (find_optimal_k, apply_kmeans,
                          visualize_clusters, analyze_segments)
from reinforcement_learning import (train_q_learning, plot_rl_training,
                                    analyze_q_table, create_final_comparison)
import numpy as np

def main():
    print("="*60)
    print(" CONSUMER BEHAVIOR PREDICTION & PRECISION MARKETING")
    print("="*60)

    # ── Phase 1: Preprocessing ──────────────────────────
    print("\n PHASE 1: DATA PREPROCESSING")
    print("-"*40)
    df = load_data()
    plot_correlation_matrix(df)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)

    # ── Phase 2: ML Models ──────────────────────────────
    print("\n PHASE 2: MACHINE LEARNING MODELS")
    print("-"*40)
    models = get_models()
    results, comparison_df = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, feature_names
    )

    # ── Phase 3: Deep Learning ──────────────────────────
    print("\n PHASE 3: DEEP LEARNING")
    print("-"*40)
    model, train_features, test_features, dl_auc, dl_acc = train_deep_learning(
        X_train, X_test, y_train, y_test
    )

    # ── Phase 4: Segmentation ───────────────────────────
    print("\n👥 PHASE 4: BEHAVIORAL SEGMENTATION")
    print("-"*40)
    optimal_k = find_optimal_k(train_features)
    kmeans, labels = apply_kmeans(train_features, optimal_k)
    visualize_clusters(train_features, labels, optimal_k)
    y_train_np = np.array(y_train)
    segment_df = analyze_segments(train_features, labels, y_train_np, optimal_k)

    # ── Phase 5: Reinforcement Learning ─────────────────
    print("\n🎮 PHASE 5: REINFORCEMENT LEARNING")
    print("-"*40)
    agent, rewards, epsilon_hist = train_q_learning(
        labels, y_train_np, n_episodes=1000
    )
    plot_rl_training(rewards, epsilon_hist)
    results_df = analyze_q_table(agent, optimal_k)
    final_df = create_final_comparison()

    # ── Done ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PROJECT COMPLETE!")
    print("  All outputs saved in outputs/")
    print("   ├── figures/  → All plots")
    print("   └── reports/  → All reports & CSVs")
    print("="*60)

if __name__ == "__main__":
    main()