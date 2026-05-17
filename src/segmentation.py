import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import FIGURES_PATH, REPORTS_PATH

# ── Find Optimal K ───────────────────────────────────────
def find_optimal_k(features, max_k=10):
    print("\n Finding Optimal Number of Clusters...")
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"   K={k} → Inertia: {kmeans.inertia_:.2f} | Silhouette: {score:.3f}")

    # Plot Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_title('Elbow Method', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].set_title('Silhouette Score', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/optimal_k_selection.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print(" Optimal K plot saved!")

    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n Optimal K = {optimal_k}")
    return optimal_k

# ── Apply KMeans ─────────────────────────────────────────
def apply_kmeans(features, k):
    print(f"\n Applying K-means with K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    return kmeans, labels

# ── Visualize Clusters ───────────────────────────────────
def visualize_clusters(features, labels, k):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(10, 7))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i in range(k):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=colors[i], label=f'Segment {i+1}',
                   alpha=0.6, s=30)

    plt.title('Customer Behavioral Segments (PCA Visualization)',
              fontsize=14, fontweight='bold')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/customer_segments.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print(" Customer Segments plot saved!")

# ── Analyze Segments ─────────────────────────────────────
def analyze_segments(features, labels, y_train, k):
    print(f"\n Segment Analysis:")
    print("="*50)

    segment_data = []
    for i in range(k):
        mask = labels == i
        segment_y = y_train[mask]
        purchase_rate = segment_y.mean() * 100
        size = mask.sum()
        segment_data.append({
            'Segment': f'Segment {i+1}',
            'Size': size,
            'Purchase Rate (%)': round(purchase_rate, 2),
            'Marketing Action': get_marketing_action(purchase_rate)
        })
        print(f"\n Segment {i+1}:")
        print(f"   Size: {size} customers")
        print(f"   Purchase Rate: {purchase_rate:.2f}%")
        print(f"   Action: {get_marketing_action(purchase_rate)}")

    segment_df = pd.DataFrame(segment_data)
    segment_df.to_csv(f"{REPORTS_PATH}/segment_analysis.csv", index=False)
    print(f"\n Segment Analysis saved!")

    # Plot Purchase Rate per Segment
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B' if r < 40 else '#4ECDC4' if r < 70 else '#45B7D1'
              for r in segment_df['Purchase Rate (%)']]
    bars = plt.bar(segment_df['Segment'], segment_df['Purchase Rate (%)'],
                   color=colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, segment_df['Purchase Rate (%)']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.title('Purchase Rate by Customer Segment', fontsize=14, fontweight='bold')
    plt.xlabel('Segment')
    plt.ylabel('Purchase Rate (%)')
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/segment_purchase_rates.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print(" Purchase Rate plot saved!")

    return segment_df

# ── Marketing Action per Segment ─────────────────────────
def get_marketing_action(purchase_rate):
    if purchase_rate >= 70:
        return " VIP - Loyalty Rewards & Premium Offers"
    elif purchase_rate >= 40:
        return " Warm - Personalized Discounts & Recommendations"
    else:
        return " Cold - Re-engagement Campaign & Special Deals"

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(" Loading Deep Features...")
    train_features = np.load(f"{REPORTS_PATH}/deep_features_train.npy")
    y_train = np.load(f"{REPORTS_PATH}/y_train.npy")

    print(f" Features Shape: {train_features.shape}")

    # Find optimal K
    optimal_k = find_optimal_k(train_features)

    # Apply KMeans
    kmeans, labels = apply_kmeans(train_features, optimal_k)

    # Visualize
    visualize_clusters(train_features, labels, optimal_k)

    # Analyze
    segment_df = analyze_segments(train_features, labels, y_train, optimal_k)

    # Save model labels
    np.save(f"{REPORTS_PATH}/cluster_labels_train.npy", labels)
    np.save(f"{REPORTS_PATH}/kmeans_model.npy",
            np.array([kmeans], dtype=object))

    print("\n" + "="*50)
    print(" Behavioral Segmentation Complete!")
    print(f"   {optimal_k} Customer Segments Identified")
    print("="*50)