import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys
import argparse


def print_log(msg):
    print(f"[+] {msg}")


def harmonize_features(df_unsw, df_cic):
    print_log("Harmonizing Features between UNSW and CIC datasets...")
    # Mapping UNSW column names to CICFlowMeter common column names
    feature_mapping = {
        'dur': 'Flow Duration',
        'proto': 'Protocol',
        'spkts': 'Total Fwd Packets',
        'dpkts': 'Total Backward Packets',
        'sbytes': 'Total Length of Fwd Packets',
        'dbytes': 'Total Length of Bwd Packets',
        'sload': 'Fwd Bytes/Bulk Avg',  # Proxy map for overlapping network concepts
        'dload': 'Bwd Bytes/Bulk Avg',
        'sloss': 'Fwd Packet Length Std',
        'dloss': 'Bwd Packet Length Std',
        'smean': 'Fwd Packet Length Mean',
        'dmean': 'Bwd Packet Length Mean'
    }

    df_unsw_mapped = df_unsw.rename(columns=feature_mapping)

    # Align labels
    if 'label' in df_unsw_mapped.columns:
        df_unsw_mapped = df_unsw_mapped.rename(columns={'label': 'target_label'})

    # CIC datasets often use 'Label' or 'label'
    if 'Label' in df_cic.columns:
        df_cic = df_cic.rename(columns={'Label': 'target_label'})
    elif 'label' in df_cic.columns:
        df_cic = df_cic.rename(columns={'label': 'target_label'})

    # Unify target_label to 0 = Normal, 1 = Attack
    def convert_label(val):
        if pd.isna(val):
            return 0
        if isinstance(val, str):
            val = val.lower()
            return 0 if ('normal' in val or 'benign' in val) else 1
        return int(val)

    if 'target_label' in df_unsw_mapped.columns:
        df_unsw_mapped['target_label'] = df_unsw_mapped['target_label'].apply(convert_label)
    if 'target_label' in df_cic.columns:
        df_cic['target_label'] = df_cic['target_label'].apply(convert_label)

    # Identify overlapping features and drop the rest
    common_cols = list(set(df_unsw_mapped.columns).intersection(set(df_cic.columns)))
    print_log(f"Found {len(common_cols)} overlapping features after mapping. Dropping non-overlapping columns.")

    df_merged = pd.concat([df_unsw_mapped[common_cols], df_cic[common_cols]], ignore_index=True)
    return df_merged


def preprocess_and_subsample(df, target_size=300000):
    print_log("Handling Missing Values (NaN) and Infs...")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            valid_mask = ~df[col].isin([np.inf, -np.inf, np.nan])
            if valid_mask.sum() > 0:
                perc_99 = np.percentile(df.loc[valid_mask, col], 99)
                median_val = df.loc[valid_mask, col].median()
            else:
                perc_99 = 0
                median_val = 0

            # Replace infs with 99th percentile
            df[col] = df[col].replace([np.inf, -np.inf], perc_99)
            # Impute standard NaNs with median
            df[col] = df[col].fillna(median_val)
        else:
            # Handle categorical NaNs
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    print_log("Encoding Categorical Features...")
    for col in df.columns:
        if col != 'target_label' and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    if len(df) > target_size:
        print_log(f"Sub-sampling merged dataset to {target_size} rows (Stratified)...")
        df, _ = train_test_split(df, train_size=target_size, stratify=df['target_label'], random_state=42)

    print_log(f"Final merged dataset shape: {df.shape}")
    return df


def print_ascii_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # Ensure correct 2x2 unpacking in case of singular labels
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        print_log("Warning: Matrix is not 2x2. Model might be predicting only one class.")

    print("\n" + "=" * 45)
    print("           ASCII CONFUSION MATRIX")
    print("=" * 45)
    print("                       Predicted")
    print("                 0 (Normal)    1 (Attack)")
    print("                +-------------+-------------+")
    print(f"Actual 0 (Nrml) | {tn:<11} | {fp:<11} |")
    print(f"       1 (Atk)  | {fn:<11} | {tp:<11} |")
    print("                +-------------+-------------+\n")


def main():
    # Force the script to look in the exact folder where it is saved
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="Network IDS ML Pipeline")
    # FIX 1: Default is now explicitly .csv
    parser.add_argument('--unsw', type=str, default='UNSW_NB15_training-set.csv', help='Path to UNSW dataset')
    parser.add_argument('--cic', type=str, default='CIC-BoT-IoT-V2.parquet', help='Path to CIC dataset')
    args = parser.parse_args()

    print_log("Initializing ML Pipeline for Network IDS")

    unsw_path = args.unsw
    cic_path = args.cic

    if not os.path.exists(unsw_path) or not os.path.exists(cic_path):
        print_log(f"WARNING: Datasets not found.")
        print_log(f"Please ensure '{unsw_path}' and '{cic_path}' are in: {script_dir}")
        sys.exit(1)

    print_log("Loading Data...")

    # FIX 2: Use the unsw_path variable, no openpyxl check needed for CSV
    df_unsw = pd.read_csv(unsw_path)

    try:
        df_cic = pd.read_parquet(cic_path, engine='pyarrow')
    except ImportError:
        print_log("pyarrow missing, falling back to fastparquet engine.")
        df_cic = pd.read_parquet(cic_path, engine='fastparquet')

    # 1. Harmonize Data
    df_merged = harmonize_features(df_unsw, df_cic)

    # 2. Preprocess & Sub-sample Data
    df_final = preprocess_and_subsample(df_merged, target_size=300000)

    # 3. Train/Test Split
    print_log("Splitting Data (80/20 train/test split)...")
    X = df_final.drop('target_label', axis=1)
    y = df_final['target_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 4. Normalize
    print_log("Normalizing numerical features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. Train Model
    print_log("Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # 6. Evaluate
    print_log("Evaluating Model...")
    y_pred = rf.predict(X_test)

    print("\n" + "=" * 45)
    print("           CLASSIFICATION REPORT")
    print("=" * 45)
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))

    print_ascii_confusion_matrix(y_test, y_pred)

    # 7. Save Model
    model_filename = '.venv/network_ids_model.pkl'
    print_log(f"Saving trained model to {model_filename}...")
    joblib.dump(rf, model_filename)
    print_log("Pipeline Complete.")


if __name__ == '__main__':
    main()
