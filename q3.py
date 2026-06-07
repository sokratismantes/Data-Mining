import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

INPUT_FILE = "cic_ids_2017_ml_ready.parquet"
OUTPUT_DIR = "question_3_outputs"

RANDOM_STATE = 42

MAIN_SAMPLE_SIZE = 30000

HIERARCHICAL_SAMPLE_SIZE = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_filename(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(",", "_")
    )

def evaluate_clustering(X_data, clusters):

    unique_clusters = set(clusters)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    noise_points = int(np.sum(clusters == -1))
    noise_ratio = float(noise_points / len(clusters) * 100)

    if n_clusters <= 1:
        return {
            "Silhouette": np.nan,
            "Davies_Bouldin": np.nan,
            "Number_of_Clusters": n_clusters,
            "Noise_Points": noise_points,
            "Noise_Ratio_Percentage": noise_ratio,
            "Valid": False
        }

    sil = silhouette_score(X_data, clusters)
    db = davies_bouldin_score(X_data, clusters)

    return {
        "Silhouette": sil,
        "Davies_Bouldin": db,
        "Number_of_Clusters": n_clusters,
        "Noise_Points": noise_points,
        "Noise_Ratio_Percentage": noise_ratio,
        "Valid": True
    }

def save_cluster_vs_label(y_labels, clusters, algorithm_name):

    table = pd.crosstab(
        pd.Series(y_labels, name="True Label"),
        pd.Series(clusters, name="Cluster")
    )
    filename = f"cluster_vs_label_{clean_filename(algorithm_name)}.csv"
    table.to_csv(os.path.join(OUTPUT_DIR, filename))
    return table

def save_pca_plot(X_pca, clusters, title, filename, label_name="Cluster"):

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=clusters,
        s=5,
        alpha=0.6,
        cmap="tab20"
    )
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label=label_name)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()

print("============================================================")
print("ΕΡΩΤΗΜΑ 3 - CLUSTERING")
print("============================================================")

print("\n[1] Φόρτωση ML-ready dataset...")

df = pd.read_parquet(INPUT_FILE)

if "Label" not in df.columns:
    raise ValueError("Δεν βρέθηκε η στήλη 'Label' στο dataset.")

X_full = df.drop(columns=["Label"])
y_full = df["Label"].reset_index(drop=True)

print("\n[2] Δημιουργία δείγματος για clustering...")

np.random.seed(RANDOM_STATE)

main_sample_size = min(MAIN_SAMPLE_SIZE, len(df))
main_indices = np.random.choice(len(df), size=main_sample_size, replace=False)

X_sample = X_full.iloc[main_indices].reset_index(drop=True)
y_sample = y_full.iloc[main_indices].reset_index(drop=True)

print(f"Κύριο δείγμα: {X_sample.shape}")

hier_sample_size = min(HIERARCHICAL_SAMPLE_SIZE, len(X_sample))
hier_indices = np.random.choice(len(X_sample), size=hier_sample_size, replace=False)

X_hier_sample = X_sample.iloc[hier_indices].reset_index(drop=True)
y_hier_sample = y_sample.iloc[hier_indices].reset_index(drop=True)

print(f"Δείγμα Hierarchical: {X_hier_sample.shape}")

del df, X_full, y_full

print("\n[3] StandardScaler...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample).astype(np.float32)

X_hier_scaled = scaler.transform(X_hier_sample).astype(np.float32)

print("\n[4] PCA για οπτικοποίηση...")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

pca_info = pd.DataFrame({
    "Component": ["PCA 1", "PCA 2"],
    "Explained Variance Ratio": pca.explained_variance_ratio_
})
pca_info.to_csv(os.path.join(OUTPUT_DIR, "pca_explained_variance.csv"), index=False)

label_codes, label_uniques = pd.factorize(y_sample)
label_mapping = pd.DataFrame({
    "Code": range(len(label_uniques)),
    "Label": label_uniques
})
label_mapping.to_csv(os.path.join(OUTPUT_DIR, "pca_label_code_mapping.csv"), index=False)

save_pca_plot(
    X_pca,
    label_codes,
    "Πραγματικά Labels στον χώρο PCA",
    "pca_true_labels.png",
    label_name="True Label Code"
)

pca_hier = PCA(n_components=2, random_state=RANDOM_STATE)
X_hier_pca = pca_hier.fit_transform(X_hier_scaled)

print("\n[5] K-Means για διαφορετικά k...")

results = []
stored_clusterings = {}

k_values = [2, 3, 4, 5, 6, 8, 10]

inertias = []

for k in k_values:
    algorithm_name = f"KMeans_k_{k}"
    print(f"Εκτέλεση {algorithm_name}...")

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)

    metrics = evaluate_clustering(X_scaled, clusters)

    results.append({
        "Algorithm": "K-Means",
        "Parameters": f"k={k}",
        **metrics
    })

    save_cluster_vs_label(y_sample, clusters, algorithm_name)

    stored_clusterings[algorithm_name] = {
        "clusters": clusters,
        "metrics": metrics,
        "sample_type": "main"
    }

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.title("K-Means Elbow Method")
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_elbow_method.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\n[6] Hierarchical Clustering με διαφορετικά linkages...")

linkages = ["ward", "average", "complete"]
hier_k_values = [3, 5]

for linkage in linkages:
    for k in hier_k_values:
        algorithm_name = f"Hierarchical_{linkage}_k_{k}"
        print(f"Εκτέλεση {algorithm_name}...")

        try:
            hierarchical = AgglomerativeClustering(
                n_clusters=k,
                linkage=linkage
            )

            clusters = hierarchical.fit_predict(X_hier_scaled)

            metrics = evaluate_clustering(X_hier_scaled, clusters)

            results.append({
                "Algorithm": "Hierarchical",
                "Parameters": f"linkage={linkage}, k={k}",
                **metrics
            })

            save_cluster_vs_label(y_hier_sample, clusters, algorithm_name)

            stored_clusterings[algorithm_name] = {
                "clusters": clusters,
                "metrics": metrics,
                "sample_type": "hierarchical"
            }

        except Exception as e:
            print(f"Σφάλμα στο {algorithm_name}: {e}")

            results.append({
                "Algorithm": "Hierarchical",
                "Parameters": f"linkage={linkage}, k={k}",
                "Silhouette": np.nan,
                "Davies_Bouldin": np.nan,
                "Number_of_Clusters": np.nan,
                "Noise_Points": np.nan,
                "Noise_Ratio_Percentage": np.nan,
                "Valid": False
            })

print("\n[7] DBSCAN με διαφορετικά eps/min_samples...")

dbscan_params = [
    (1.0, 5),
    (1.5, 10),
    (2.0, 10),
    (2.0, 15),
    (3.0, 20)
]

for eps, min_samples in dbscan_params:
    algorithm_name = f"DBSCAN_eps_{eps}_min_samples_{min_samples}"
    print(f"Εκτέλεση {algorithm_name}...")

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=2
    )

    clusters = dbscan.fit_predict(X_scaled)

    metrics = evaluate_clustering(X_scaled, clusters)

    results.append({
        "Algorithm": "DBSCAN",
        "Parameters": f"eps={eps}, min_samples={min_samples}",
        **metrics
    })

    save_cluster_vs_label(y_sample, clusters, algorithm_name)

    stored_clusterings[algorithm_name] = {
        "clusters": clusters,
        "metrics": metrics,
        "sample_type": "main"
    }

    print(
        f"Clusters: {metrics['Number_of_Clusters']}, "
        f"Noise: {metrics['Noise_Ratio_Percentage']:.2f}%"
    )

print("\n[8] Αποθήκευση συγκριτικού πίνακα clustering...")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "clustering_metrics_results.csv"), index=False)

print(results_df.to_string(index=False))

print("\n[9] PCA plots για τα καλύτερα μοντέλα...")

valid_results = results_df[results_df["Valid"] == True].copy()

if len(valid_results) > 0:

    best_silhouette_row = valid_results.sort_values("Silhouette", ascending=False).iloc[0]

    best_db_row = valid_results.sort_values("Davies_Bouldin", ascending=True).iloc[0]

    best_per_algorithm = (
        valid_results
        .sort_values("Silhouette", ascending=False)
        .groupby("Algorithm")
        .head(1)
    )

    best_per_algorithm.to_csv(os.path.join(OUTPUT_DIR, "best_clustering_per_algorithm.csv"), index=False)

    for _, row in best_per_algorithm.iterrows():
        alg = row["Algorithm"]
        params = row["Parameters"]

        if alg == "K-Means":

            k = params.split("=")[1]
            key = f"KMeans_k_{k}"

        elif alg == "Hierarchical":

            parts = params.replace(" ", "").split(",")
            linkage = parts[0].split("=")[1]
            k = parts[1].split("=")[1]
            key = f"Hierarchical_{linkage}_k_{k}"

        elif alg == "DBSCAN":

            clean = params.replace(" ", "")
            eps_part, ms_part = clean.split(",")
            eps = eps_part.split("=")[1]
            ms = ms_part.split("=")[1]
            key = f"DBSCAN_eps_{eps}_min_samples_{ms}"

        else:
            continue

        if key in stored_clusterings:
            clusters = stored_clusterings[key]["clusters"]
            sample_type = stored_clusterings[key]["sample_type"]

            if sample_type == "main":
                plot_data = X_pca
            else:
                plot_data = X_hier_pca

            save_pca_plot(
                plot_data,
                clusters,
                f"PCA Clustering - {alg} ({params})",
                f"pca_best_{clean_filename(alg)}.png",
                label_name="Cluster"
            )

    conclusion = []
    conclusion.append("ΣΥΜΠΕΡΑΣΜΑ CLUSTERING\n")
    conclusion.append("============================\n")
    conclusion.append(f"Καλύτερη μέθοδος ως προς Silhouette:\n")
    conclusion.append(f"- Algorithm: {best_silhouette_row['Algorithm']}\n")
    conclusion.append(f"- Parameters: {best_silhouette_row['Parameters']}\n")
    conclusion.append(f"- Silhouette: {best_silhouette_row['Silhouette']:.4f}\n")
    conclusion.append(f"- Davies-Bouldin: {best_silhouette_row['Davies_Bouldin']:.4f}\n")
    conclusion.append(f"- Clusters: {best_silhouette_row['Number_of_Clusters']}\n")
    conclusion.append(f"- Noise ratio: {best_silhouette_row['Noise_Ratio_Percentage']:.2f}%\n\n")

    conclusion.append(f"Καλύτερη μέθοδος ως προς Davies-Bouldin:\n")
    conclusion.append(f"- Algorithm: {best_db_row['Algorithm']}\n")
    conclusion.append(f"- Parameters: {best_db_row['Parameters']}\n")
    conclusion.append(f"- Silhouette: {best_db_row['Silhouette']:.4f}\n")
    conclusion.append(f"- Davies-Bouldin: {best_db_row['Davies_Bouldin']:.4f}\n")
    conclusion.append(f"- Clusters: {best_db_row['Number_of_Clusters']}\n")
    conclusion.append(f"- Noise ratio: {best_db_row['Noise_Ratio_Percentage']:.2f}%\n\n")

    conclusion.append("Σχόλιο για τη σύγκριση με Label:\n")
    conclusion.append(
        "Για κάθε εκτέλεση έχει αποθηκευτεί αρχείο cluster_vs_label_*.csv. "
        "Αυτοί οι πίνακες δείχνουν πώς κατανέμονται οι πραγματικές κλάσεις Label μέσα στα clusters. "
        "Επειδή το clustering είναι μη επιβλεπόμενη μέθοδος, το Label δεν χρησιμοποιείται στην εκπαίδευση, "
        "αλλά μόνο εκ των υστέρων για ερμηνεία.\n\n"
    )

    conclusion.append("Σχόλιο για περιορισμούς υπολογιστή:\n")
    conclusion.append(
        f"Χρησιμοποιήθηκε κύριο δείγμα {MAIN_SAMPLE_SIZE} γραμμών και μικρότερο δείγμα "
        f"{HIERARCHICAL_SAMPLE_SIZE} γραμμών για Hierarchical Clustering, επειδή ο συγκεκριμένος "
        "αλγόριθμος είναι υπολογιστικά βαρύς σε μεγάλο αριθμό παρατηρήσεων.\n"
    )

    with open(os.path.join(OUTPUT_DIR, "clustering_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write("".join(conclusion))

else:
    with open(os.path.join(OUTPUT_DIR, "clustering_conclusion.txt"), "w", encoding="utf-8") as f:
        f.write("Δεν βρέθηκε έγκυρο clustering με περισσότερες από μία ομάδες.\n")

print("\n============================================================")
print("ΟΛΟΚΛΗΡΩΣΗ ΕΡΩΤΗΜΑΤΟΣ 3")
print("============================================================")
print(f"Όλα τα αποτελέσματα αποθηκεύτηκαν στον φάκελο: {OUTPUT_DIR}")
