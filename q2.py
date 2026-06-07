import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sns.set_theme(style="whitegrid")

print("Φόρτωση του ML-ready συνόλου δεδομένων...")
df = pd.read_parquet('cic_ids_2017_ml_ready.parquet')

X = df.drop(columns=['Label'])

y_raw = df['Label'].astype(str).str.replace("�", "", regex=False)

le = LabelEncoder()
y = le.fit_transform(y_raw)

all_labels = np.arange(len(le.classes_))

label_mapping = pd.DataFrame({
    "Encoded_Label": range(len(le.classes_)),
    "Original_Label": le.classes_
})
label_mapping.to_csv("label_encoding_mapping.csv", index=False)

print("\nΕφαρμογή Στρωματοποιημένης Δειγματοληψίας (20%) λόγω περιορισμών μνήμης...")
X_sampled, _, y_sampled, _ = train_test_split(
    X, y, train_size=0.20, stratify=y, random_state=42
)

full_distribution = pd.Series(y_raw).value_counts(normalize=True).sort_index() * 100
sample_distribution = pd.Series(le.inverse_transform(y_sampled)).value_counts(normalize=True).sort_index() * 100

distribution_df = pd.DataFrame({
    "Full_Dataset_Percentage": full_distribution,
    "Sampled_Dataset_Percentage": sample_distribution
})
distribution_df.to_csv("label_distribution_full_vs_sampled.csv")

del df, X, y, y_raw

print("Διαχωρισμός του δείγματος σε Train/Val και Test σύνολα...")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_sampled, y_sampled, test_size=0.20, stratify=y_sampled, random_state=42
)

del X_sampled, y_sampled

print("Εφαρμογή κανονικοποίησης (StandardScaler)...")
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

del X_train_val, X_test

print(f"Διαστάσεις συνόλου Εκπαίδευσης/Βελτιστοποίησης: {X_train_val_scaled.shape}")
print(f"Διαστάσεις συνόλου Ελέγχου (Test): {X_test_scaled.shape}")

print("\nΠροετοιμασία δομών για τα σενάρια βελτιστοποίησης...")

X_train_only, X_val_only, y_train_only, y_val_only = train_test_split(
    X_train_val_scaled, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
)

split_index = [-1] * len(y_train_only) + [0] * len(y_val_only)
X_scenario_a = np.vstack((X_train_only, X_val_only)).astype(np.float32)
y_scenario_a = np.concatenate((y_train_only, y_val_only))
scenario_a_split = PredefinedSplit(test_fold=split_index)

del X_train_only, X_val_only, y_train_only, y_val_only

scenario_b_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

models_and_params = {
    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=500,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        ),
        "params": {
            'C': [0.1, 10.0]
        }
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(
            class_weight='balanced',
            random_state=42
        ),
        "params": {
            'max_depth': [10, 20],
            'min_samples_split': [2, 10]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            class_weight='balanced',
            n_jobs=2,
            random_state=42
        ),
        "params": {
            'n_estimators': [30, 60],
            'max_depth': [12, 18]
        }
    }
}

def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")

def evaluate_model(y_true, y_pred, model_name, scenario_name):

    acc = accuracy_score(y_true, y_pred)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n--- Αποτελέσματα: {model_name} ({scenario_name}) ---")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Precision weighted: {prec_weighted:.4f}")
    print(f"Recall weighted:    {rec_weighted:.4f}")
    print(f"F1 weighted:        {f1_weighted:.4f}")
    print(f"Precision macro:    {prec_macro:.4f}")
    print(f"Recall macro:       {rec_macro:.4f}")
    print(f"F1 macro:           {f1_macro:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=le.classes_,
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(
        f"classification_report_{safe_name(model_name)}_{safe_name(scenario_name)}.csv",
        index=True
    )

    return acc, prec_weighted, rec_weighted, f1_weighted, prec_macro, rec_macro, f1_macro

def plot_confusion_matrix(y_true, y_pred, model_name, scenario_name):

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(f"cm_{safe_name(model_name)}_{safe_name(scenario_name)}.csv")

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} ({scenario_name})')
    plt.ylabel('Πραγματική Κλάση')
    plt.xlabel('Προβλεπόμενη Κλάση')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"cm_{safe_name(model_name)}_{safe_name(scenario_name)}.png", dpi=300, bbox_inches="tight")
    plt.close()

    cm_norm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize="true")
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    cm_norm_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
    cm_norm_df.to_csv(f"cm_normalized_{safe_name(model_name)}_{safe_name(scenario_name)}.csv")

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm_df, annot=False, cmap='Blues', vmin=0, vmax=1)
    plt.title(f'Normalized Confusion Matrix: {model_name} ({scenario_name})')
    plt.ylabel('Πραγματική Κλάση')
    plt.xlabel('Προβλεπόμενη Κλάση')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"cm_normalized_{safe_name(model_name)}_{safe_name(scenario_name)}.png", dpi=300, bbox_inches="tight")
    plt.close()

results_records = []

for model_name, mp in models_and_params.items():
    print(f"\n==================================================")
    print(f"Ξεκινάει η εκπαίδευση για: {model_name}")
    print(f"==================================================")

    print(f"--> Εκτέλεση Grid Search (Σενάριο Α: Train/Val Split)...")

    grid_a = GridSearchCV(
        estimator=mp["model"],
        param_grid=mp["params"],
        scoring='f1_weighted',
        cv=scenario_a_split,
        n_jobs=1
    )

    grid_a.fit(X_scenario_a, y_scenario_a)

    pd.DataFrame(grid_a.cv_results_).to_csv(
        f"grid_results_{safe_name(model_name)}_Train_Val_Split.csv",
        index=False
    )

    print(f"Καλύτερες Υπερπαράμετροι (Σενάριο Α): {grid_a.best_params_}")

    best_model_a = grid_a.best_estimator_
    preds_a = best_model_a.predict(X_test_scaled)

    acc_a, pw_a, rw_a, f1w_a, pm_a, rm_a, f1m_a = evaluate_model(
        y_test, preds_a, model_name, "Train_Val_Split"
    )
    plot_confusion_matrix(y_test, preds_a, model_name, "Train_Val_Split")

    results_records.append({
        "Model": model_name,
        "Optimization": "Train/Val Split",
        "Best Params": str(grid_a.best_params_),
        "Accuracy": acc_a,
        "Precision_weighted": pw_a,
        "Recall_weighted": rw_a,
        "F1_weighted": f1w_a,
        "Precision_macro": pm_a,
        "Recall_macro": rm_a,
        "F1_macro": f1m_a
    })

    print(f"\n--> Εκτέλεση Grid Search (Σενάριο Β: 2-Fold CV)...")

    grid_b = GridSearchCV(
        estimator=mp["model"],
        param_grid=mp["params"],
        scoring='f1_weighted',
        cv=scenario_b_cv,
        n_jobs=1
    )

    grid_b.fit(X_train_val_scaled, y_train_val)

    pd.DataFrame(grid_b.cv_results_).to_csv(
        f"grid_results_{safe_name(model_name)}_Cross_Validation.csv",
        index=False
    )

    print(f"Καλύτερες Υπερπαράμετροι (Σενάριο Β): {grid_b.best_params_}")

    best_model_b = grid_b.best_estimator_
    preds_b = best_model_b.predict(X_test_scaled)

    acc_b, pw_b, rw_b, f1w_b, pm_b, rm_b, f1m_b = evaluate_model(
        y_test, preds_b, model_name, "Cross_Validation"
    )
    plot_confusion_matrix(y_test, preds_b, model_name, "Cross_Validation")

    results_records.append({
        "Model": model_name,
        "Optimization": "Cross-Validation",
        "Best Params": str(grid_b.best_params_),
        "Accuracy": acc_b,
        "Precision_weighted": pw_b,
        "Recall_weighted": rw_b,
        "F1_weighted": f1w_b,
        "Precision_macro": pm_b,
        "Recall_macro": rm_b,
        "F1_macro": f1m_b
    })

results_df = pd.DataFrame(results_records)

print("\n=======================================================================")
print("                    ΤΕΛΙΚΟΣ ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ")
print("=======================================================================")
print(results_df.to_string(index=False))

results_df.to_csv('model_comparison_results.csv', index=False)

best_row = results_df.sort_values("F1_weighted", ascending=False).iloc[0]

with open("classification_conclusion.txt", "w", encoding="utf-8") as f:
    f.write("ΣΥΜΠΕΡΑΣΜΑ ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗΣ\n")
    f.write("============================\n")
    f.write(f"Καλύτερο μοντέλο βάσει F1_weighted: {best_row['Model']}\n")
    f.write(f"Μέθοδος βελτιστοποίησης: {best_row['Optimization']}\n")
    f.write(f"Best params: {best_row['Best Params']}\n")
    f.write(f"Accuracy: {best_row['Accuracy']:.4f}\n")
    f.write(f"Precision weighted: {best_row['Precision_weighted']:.4f}\n")
    f.write(f"Recall weighted: {best_row['Recall_weighted']:.4f}\n")
    f.write(f"F1 weighted: {best_row['F1_weighted']:.4f}\n")
    f.write(f"F1 macro: {best_row['F1_macro']:.4f}\n")
    f.write("\n")
    f.write("Το F1_weighted χρησιμοποιείται για τη συνολική σύγκριση, ενώ το F1_macro ")
    f.write("σχολιάζεται επειδή το dataset είναι ανισόρροπο και περιλαμβάνει σπάνιες κλάσεις.\n")

print("\nΤα αποτελέσματα αποθηκεύτηκαν στο 'model_comparison_results.csv'.")
print("Το τελικό συμπέρασμα αποθηκεύτηκε στο 'classification_conclusion.txt'.")
