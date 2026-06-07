import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

INPUT_FILE = "cic_ids_2017_optimized.parquet"
OUTPUT_DIR = "question_1_outputs"
ML_READY_FILE = "cic_ids_2017_ml_ready.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_text(filename, text):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def safe_feature_list(df, desired_features):

    return [col for col in desired_features if col in df.columns]

print("============================================================")
print("ΕΡΩΤΗΜΑ 1 - EDA & FEATURE SELECTION")
print("============================================================")

print("\n[1] Φόρτωση dataset...")
df = pd.read_parquet(INPUT_FILE)
df.columns = df.columns.str.strip()

structure_text = []
structure_text.append("ΔΟΜΗ DATASET\n")
structure_text.append("============================\n")
structure_text.append(f"Αριθμός γραμμών: {df.shape[0]}\n")
structure_text.append(f"Αριθμός στηλών: {df.shape[1]}\n\n")
structure_text.append("Τύποι δεδομένων ανά στήλη:\n")
structure_text.append(str(df.dtypes))
structure_text.append("\n")

print(f"Γραμμές: {df.shape[0]}")
print(f"Στήλες: {df.shape[1]}")

save_text("dataset_structure.txt", "".join(structure_text))

dtypes_df = df.dtypes.reset_index()
dtypes_df.columns = ["Feature", "Dtype"]
dtypes_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_dtypes.csv"), index=False)

print("\n[2] Έλεγχος missing values και duplicates...")

missing_per_column = df.isnull().sum().sort_values(ascending=False)
missing_df = pd.DataFrame({
    "Feature": missing_per_column.index,
    "Missing Values": missing_per_column.values,
    "Missing Percentage": (missing_per_column.values / len(df)) * 100
})
missing_df.to_csv(os.path.join(OUTPUT_DIR, "missing_values_per_column.csv"), index=False)

duplicates_count = df.duplicated().sum()

quality_text = []
quality_text.append("ΕΛΕΓΧΟΣ ΠΟΙΟΤΗΤΑΣ ΔΕΔΟΜΕΝΩΝ\n")
quality_text.append("============================\n")
quality_text.append(f"Συνολικές missing values: {df.isnull().sum().sum()}\n")
quality_text.append(f"Διπλότυπες εγγραφές: {duplicates_count}\n\n")
quality_text.append("Σημείωση: Το αρχικό καθάρισμα έχει γίνει στο to_parquet.py.\n")
quality_text.append("Εδώ καταγράφουμε την κατάσταση του dataset που χρησιμοποιείται στην ανάλυση.\n")
save_text("data_quality_summary.txt", "".join(quality_text))

print(f"Συνολικές missing values: {df.isnull().sum().sum()}")
print(f"Διπλότυπες εγγραφές: {duplicates_count}")

print("\n[3] Ανάλυση μεταβλητής στόχου Label...")

if "Label" not in df.columns:
    raise ValueError("Δεν βρέθηκε η στήλη 'Label' στο dataset.")

label_counts = df["Label"].value_counts()
label_percentages = df["Label"].value_counts(normalize=True) * 100

label_summary = pd.DataFrame({
    "Count": label_counts,
    "Percentage": label_percentages
})
label_summary.to_csv(os.path.join(OUTPUT_DIR, "label_distribution_summary.csv"))

print(label_summary)

plt.figure(figsize=(12, 7))
sns.barplot(
    x=label_summary["Count"].values,
    y=label_summary.index.astype(str),
    hue=label_summary.index.astype(str),
    palette="viridis",
    legend=False
)
plt.xscale("log")
plt.title("Κατανομή κλάσεων Label - λογαριθμική κλίμακα")
plt.xlabel("Αριθμός εγγραφών (log scale)")
plt.ylabel("Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\n[4] Υπολογισμός στατιστικών για όλες τις αριθμητικές στήλες...")

df_numeric = df.select_dtypes(include=[np.number])

descriptive_stats = df_numeric.describe().T
descriptive_stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics_all_features.csv"))

print("Τα πλήρη στατιστικά αποθηκεύτηκαν στο descriptive_statistics_all_features.csv")

extra_stats = pd.DataFrame({
    "Variance": df_numeric.var(numeric_only=True),
    "Skewness": df_numeric.skew(numeric_only=True),
    "Kurtosis": df_numeric.kurtosis(numeric_only=True)
})
extra_stats.to_csv(os.path.join(OUTPUT_DIR, "extra_statistics_variance_skewness_kurtosis.csv"))

print("\n[5] Δημιουργία γραφημάτων...")

candidate_features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Average Packet Size"
]

features_to_plot = safe_feature_list(df, candidate_features)

if len(features_to_plot) == 0:
    features_to_plot = list(df_numeric.columns[:6])

plot_sample_size = min(50000, len(df))
df_plot = df.sample(n=plot_sample_size, random_state=42) if len(df) > plot_sample_size else df.copy()

for feature in features_to_plot[:6]:
    plt.figure(figsize=(9, 5))
    values = df_plot[feature].replace([np.inf, -np.inf], np.nan).dropna()

    sns.histplot(np.log1p(values.clip(lower=0)), bins=50, kde=False)
    plt.title(f"Κατανομή του {feature} με log1p μετασχηματισμό")
    plt.xlabel(f"log1p({feature})")
    plt.ylabel("Συχνότητα")
    plt.tight_layout()
    safe_name = feature.replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{safe_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

boxplot_features = features_to_plot[:4]
if len(boxplot_features) > 0:
    rows = 2
    cols = 2
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(boxplot_features, 1):
        plt.subplot(rows, cols, i)
        temp = df_plot[["Label", feature]].copy()
        temp[feature] = temp[feature].replace([np.inf, -np.inf], np.nan)
        temp = temp.dropna()

        temp[feature] = np.log1p(temp[feature].clip(lower=0))
        sns.boxplot(x="Label", y=feature, data=temp)
        plt.title(f"Boxplot: log1p({feature}) ανά Label")
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplots_selected_features_by_label.png"), dpi=300, bbox_inches="tight")
    plt.close()

print("\n[6] Υπολογισμός συσχετίσεων...")

corr_matrix = df_numeric.corr().abs()
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix_all_numeric_features.csv"))

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
top_corr_pairs = upper_tri.stack().sort_values(ascending=False).reset_index()
top_corr_pairs.columns = ["Feature 1", "Feature 2", "Absolute Pearson Correlation"]
top_corr_pairs.to_csv(os.path.join(OUTPUT_DIR, "top_correlated_feature_pairs.csv"), index=False)

print("Τα top correlated feature pairs αποθηκεύτηκαν.")

heatmap_features = list(df_numeric.columns[:20])
plt.figure(figsize=(14, 11))
sns.heatmap(df_numeric[heatmap_features].corr(), cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Correlation Heatmap - πρώτα 20 αριθμητικά χαρακτηριστικά")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_first_20_features.png"), dpi=300, bbox_inches="tight")
plt.close()

mean_corr = corr_matrix.mean().sort_values(ascending=False)
top_corr_features = list(mean_corr.head(min(20, len(mean_corr))).index)
plt.figure(figsize=(14, 11))
sns.heatmap(df_numeric[top_corr_features].corr(), cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Correlation Heatmap - χαρακτηριστικά με υψηλή μέση συσχέτιση")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_top_correlated_features.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\n[7] Feature selection...")

removed_features_records = []

variance_threshold = 0.01
selector = VarianceThreshold(threshold=variance_threshold)
selector.fit(df_numeric)

low_variance_columns = [
    col for col, keep in zip(df_numeric.columns, selector.get_support()) if not keep
]

for col in low_variance_columns:
    removed_features_records.append({
        "Feature": col,
        "Reason": f"Low variance < {variance_threshold}",
        "Details": f"Variance={df_numeric[col].var()}"
    })

df_reduced = df.drop(columns=low_variance_columns)

correlation_threshold = 0.95
df_numeric_reduced = df_reduced.select_dtypes(include=[np.number])
corr_reduced = df_numeric_reduced.corr().abs()
upper_tri_reduced = corr_reduced.where(np.triu(np.ones(corr_reduced.shape), k=1).astype(bool))

highly_correlated_columns = []
high_corr_reason = {}

for column in upper_tri_reduced.columns:
    correlated_with = upper_tri_reduced.index[upper_tri_reduced[column] > correlation_threshold].tolist()
    if len(correlated_with) > 0:
        highly_correlated_columns.append(column)
        strongest_feature = upper_tri_reduced[column].idxmax()
        strongest_corr = upper_tri_reduced[column].max()
        high_corr_reason[column] = (strongest_feature, strongest_corr)

for col in highly_correlated_columns:
    other_col, corr_value = high_corr_reason[col]
    removed_features_records.append({
        "Feature": col,
        "Reason": f"High correlation > {correlation_threshold}",
        "Details": f"Highly correlated with '{other_col}', corr={corr_value:.4f}"
    })

df_final = df_reduced.drop(columns=highly_correlated_columns)

removed_df = pd.DataFrame(removed_features_records)
removed_df.to_csv(os.path.join(OUTPUT_DIR, "removed_features_report.csv"), index=False)

print(f"Στήλες χαμηλής διακύμανσης που αφαιρέθηκαν: {len(low_variance_columns)}")
print(f"Στήλες υψηλής συσχέτισης που αφαιρέθηκαν: {len(highly_correlated_columns)}")
print(f"Αρχικές στήλες: {df.shape[1]}")
print(f"Τελικές στήλες: {df_final.shape[1]}")

feature_selection_text = []
feature_selection_text.append("FEATURE SELECTION SUMMARY\n")
feature_selection_text.append("============================\n")
feature_selection_text.append(f"Αρχικός αριθμός στηλών: {df.shape[1]}\n")
feature_selection_text.append(f"Τελικός αριθμός στηλών: {df_final.shape[1]}\n")
feature_selection_text.append(f"Αφαιρέθηκαν λόγω χαμηλής διακύμανσης: {len(low_variance_columns)}\n")
feature_selection_text.append(f"Αφαιρέθηκαν λόγω υψηλής συσχέτισης: {len(highly_correlated_columns)}\n\n")
feature_selection_text.append("Κριτήρια:\n")
feature_selection_text.append(f"1. Χαμηλή διακύμανση: Variance < {variance_threshold}\n")
feature_selection_text.append(f"2. Υψηλή συσχέτιση: |Pearson correlation| > {correlation_threshold}\n")
save_text("feature_selection_summary.txt", "".join(feature_selection_text))

final_features_df = pd.DataFrame({"Feature": df_final.columns})
final_features_df.to_csv(os.path.join(OUTPUT_DIR, "final_features.csv"), index=False)

print("\n[8] Αποθήκευση τελικού dataset για τα Ερωτήματα 2 και 3...")

df_final.to_parquet(ML_READY_FILE, engine="pyarrow", index=False)

print("\n============================================================")
print("ΟΛΟΚΛΗΡΩΣΗ ΕΡΩΤΗΜΑΤΟΣ 1")
print("============================================================")
print(f"Αποτελέσματα/γραφήματα στον φάκελο: {OUTPUT_DIR}")
print(f"ML-ready dataset: {ML_READY_FILE}")
