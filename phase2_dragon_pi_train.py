import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def print_log(msg):
    print(f"[+] {msg}")


def tree_to_c(clf, feats, fn="power_ids.h"):
    t = clf.tree_

    def rec(n, d):
        p = "  " * d
        if t.feature[n] != _tree.TREE_UNDEFINED:
            f, th = feats[t.feature[n]], t.threshold[n]
            return (f"{p}if ({f} <= {th:.6g}f) {{\n{rec(t.children_left[n], d + 1)}"
                    f"{p}}} else {{\n{rec(t.children_right[n], d + 1)}{p}}}\n")
        return f"{p}return {int(clf.classes_[t.value[n][0].argmax()])};\n"

    code = (f"// Auto-generated Decision Tree for ESP32\n"
            f"int predict_power(float {', float '.join(feats)}) {{\n{rec(0, 1)}}}\n")

    with open(fn, "w") as f:
        f.write(code)
    print_log(f"Exported raw C++ logic to {fn}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    dataset_file = "dragon_pi_dataset.csv"

    if not os.path.exists(dataset_file):
        raise SystemExit(f"WARNING: Cannot find '{dataset_file}'")

    print_log("Loading Data...")
    df = pd.read_csv(dataset_file)
    df.columns = df.columns.str.lower()

    if 'label' not in df.columns:
        raise SystemExit("Missing 'label' column.")

    # We only use 'current' now, since 'power' was not in this specific dataset split
    base = ['current']
    miss = [c for c in base if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing {miss}; have {list(df.columns)}")

    print_log("Unit Sanity Check (Current Stats):")
    print(df[['current']].describe())
    print("-" * 50)

    if 'session' not in df.columns:
        df['session'] = 0

    w = 5
    print_log("Computing session-isolated rolling window features...")
    for c in base:
        for f in ('mean', 'std', 'max'):
            df[f'{c}_{f}'] = df.groupby('session')[c].transform(
                lambda s: getattr(s.rolling(w), f)()
            )

    feature_cols = [f'{c}_{f}' for c in base for f in ('mean', 'std', 'max')]

    df = df.dropna(subset=feature_cols + ['label'])

    chunk_size = max(1, len(df) // 50)
    df['block_id'] = np.arange(len(df)) // chunk_size
    is_test = df['block_id'] % 5 == 4

    train_df = df[~is_test]
    test_df = df[is_test]

    X_train, y_train = train_df[feature_cols], train_df['label']
    X_test, y_test = test_df[feature_cols], test_df['label']

    print_log("Training DecisionTreeClassifier...")
    clf = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    print("\n" + "=" * 45)
    print("           MODEL DIAGNOSTICS")
    print("=" * 45)

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    print(f"Baseline (Always guesses majority class) Acc: {dummy.score(X_test, y_test):.3f}")
    print(f"Decision Tree Acc:                            {clf.score(X_test, y_test):.3f}")
    print(f"Tree Structure: Leaves = {clf.get_n_leaves()}, Max Depth = {clf.get_depth()}\n")

    print("Feature Importances:")
    for name, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"   {name:14s} {imp:.3f}")

    print("\n" + "=" * 45)
    print("           CLASSIFICATION REPORT")
    print("=" * 45)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, labels=[0, 1],
                                target_names=['Normal (0)', 'Attack (1)'], zero_division=0))

    tree_to_c(clf, feature_cols, fn="power_ids.h")

    print("\n" + "=" * 45)
    print("           C++ FIRMWARE TEST VECTORS")
    print("=" * 45)

    sample = test_df.sample(5, random_state=1)
    for _, r in sample.iterrows():
        args = ", ".join(f"{r[f]:.6g}f" for f in feature_cols)
        row_df = pd.DataFrame([r[feature_cols].values], columns=feature_cols)
        expected_pred = int(clf.predict(row_df)[0])
        print(f"// Expect {expected_pred}")
        print(f"assert(predict_power({args}) == {expected_pred});")


if __name__ == '__main__':
    main()