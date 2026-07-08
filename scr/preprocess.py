from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE


def data_path_default():
    return Path(__file__).resolve().parents[1] / 'data' / 'WineQT.csv'


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else data_path_default()
    df = pd.read_csv(path)
    return df


def detect_outliers_consensus(df: pd.DataFrame, features):
    # IQR
    outliers_iqr_mask = pd.Series(False, index=df.index)
    for feat in features:
        Q1, Q3 = df[feat].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers_iqr_mask |= (df[feat] < Q1 - 1.5 * IQR) | (df[feat] > Q3 + 1.5 * IQR)

    # Z-score
    z = np.abs(stats.zscore(df[features]))
    outliers_zscore_mask = pd.Series((z > 3).any(axis=1), index=df.index)

    # Isolation Forest
    iso = IsolationForest(contamination=0.1, random_state=42)
    outliers_if_mask = pd.Series(iso.fit_predict(df[features]) == -1, index=df.index)

    consenso = outliers_iqr_mask & outliers_zscore_mask & outliers_if_mask
    return consenso


def preprocess(csv_path: str | Path | None = None, remove_outliers: bool = True, random_state: int = 7):
    df = load_data(csv_path)

    # Drop Id if exists
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Target binary
    df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

    # Features list (preserve notebook selection)
    features = [c for c in df.columns if c not in ['quality', 'quality_label', 'wine_type']]

    # Outliers removal by consensus
    if remove_outliers:
        consenso = detect_outliers_consensus(df, features)
        df = df[~consenso].reset_index(drop=True)

    # Feature engineering
    df['alcohol_acid_ratio'] = df['alcohol'] / (df['volatile acidity'] + 1e-6)
    df['free_so2_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)
    df['high_alcohol'] = (df['alcohol'] > 11).astype(int)

    # Prepare X, y
    x = df.drop(columns=['quality', 'quality_label'])
    y = df['quality_label']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # SMOTE (only on training set)
    smote = SMOTE(random_state=random_state)
    x_train_smote, y_train_smote = smote.fit_resample(x_train_scaled, y_train)

    return {
        'x_train_scaled': x_train_scaled,
        'x_test_scaled': x_test_scaled,
        'x_train_smote': x_train_smote,
        'y_train_smote': y_train_smote,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': x.columns.tolist(),
        'df': df,
    }


if __name__ == '__main__':
    print('Executando pré-processamento...')
    out = preprocess()
    print(f"Treino (original): {out['x_train_scaled'].shape}")
    print(f"Treino (com SMOTE): {out['x_train_smote'].shape}")
    print(f"Teste: {out['x_test_scaled'].shape}")
