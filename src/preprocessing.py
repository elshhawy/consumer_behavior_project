import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# ── Paths ──────────────────────────────────────────────
DATA_PATH = "data/online_shoppers_intention.csv"
FIGURES_PATH = "outputs/figures"
REPORTS_PATH = "outputs/reports"
os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(" Data Loaded Successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df

def explore_data(df):
    print("\n Basic Info:")
    print(df.info())
    print("\n Missing Values:")
    print(df.isnull().sum())
    print("\n Target Distribution:")
    print(df['Revenue'].value_counts())

def plot_correlation_matrix(df):
    df_plot = df.copy()
    
    le = LabelEncoder()
    df_plot['Month'] = le.fit_transform(df_plot['Month'])
    df_plot['VisitorType'] = le.fit_transform(df_plot['VisitorType'])
    df_plot['Weekend'] = df_plot['Weekend'].astype(int)
    df_plot['Revenue'] = df_plot['Revenue'].astype(int)
    
    rename_map = {
        'Administrative': 'Adm', 'Administrative_Duration': 'AdmD',
        'Informational': 'Inf', 'Informational_Duration': 'InfD',
        'ProductRelated': 'Prod', 'ProductRelated_Duration': 'ProdD',
        'BounceRates': 'BR', 'ExitRates': 'ER', 'PageValues': 'PV',
        'SpecialDay': 'SD', 'Month': 'Mth', 'OperatingSystems': 'OS',
        'Browser': 'Brws', 'Region': 'Rgn', 'TrafficType': 'TT',
        'VisitorType': 'VT', 'Weekend': 'Wknd', 'Revenue': 'Rev'
    }
    df_plot = df_plot.rename(columns=rename_map)
    
    corr = df_plot.corr().round(2)
    
    fig, ax = plt.subplots(figsize=(20, 16))
    
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 7},
        xticklabels=True,
        yticklabels=True,
        cbar_kws={
            "shrink": 0.8,
            "ticks": [-1.0, -0.8, -0.6, -0.4, -0.2, 
                       0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        },
        ax=ax
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), 
                        fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), 
                        fontsize=8, rotation=0)
    
    plt.title("Pearson Correlation Coefficient Matrix", 
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/correlation_matrix.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(" Correlation Matrix saved!")

def preprocess_data(df):
    df = df.copy()

    # Label Encoding
    le = LabelEncoder()
    df['Month'] = le.fit_transform(df['Month'])
    df['VisitorType'] = le.fit_transform(df['VisitorType'])
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)

    # Features & Target
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    print(f"\n After SMOTE - Class Distribution:")
    print(pd.Series(y_resampled).value_counts())

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.20,
        random_state=42,
        stratify=y_resampled
    )
    print(f"\n Train size: {X_train.shape}")
    print(f" Test size:  {X_test.shape}")

    return X_train, X_test, y_train, y_test, X.columns.tolist(), scaler

if __name__ == "__main__":
    df = load_data()
    explore_data(df)
    plot_correlation_matrix(df)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)