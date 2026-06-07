import pandas as pd
import numpy as np
import glob
import os

dataset_path = r"C:\Users\User\Documents\CIC-Data Mining\dataset"
csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))

df_list = []

print("Ξεκινάει η επεξεργασία αρχείων ένα προς ένα (για εξοικονόμηση RAM)...")

for file in csv_files:
    print(f"Διαβάζεται και συμπιέζεται το: {os.path.basename(file)}")

    # 1. Ανάγνωση του CSV
    temp_df = pd.read_csv(file)

    # 2. Βασικός Καθαρισμός (Άμεσα, για να φύγει ο "θόρυβος")
    temp_df.columns = temp_df.columns.str.strip()
    temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    temp_df.dropna(inplace=True)

    # 3. Downcasting (Το μυστικό για να σώσεις τη RAM)
    # Μετατροπή float64 σε float32
    float_cols = temp_df.select_dtypes(include=['float64']).columns
    temp_df[float_cols] = temp_df[float_cols].astype('float32')

    # Μετατροπή int64 σε int32
    int_cols = temp_df.select_dtypes(include=['int64']).columns
    temp_df[int_cols] = temp_df[int_cols].astype('int32')

    df_list.append(temp_df)

print("\nΌλα τα αρχεία διαβάστηκαν. Ένωση σε ένα DataFrame...")
# Συνενώνουμε τα ήδη συμπιεσμένα DataFrames
final_df = pd.concat(df_list, ignore_index=True)

# Διαγραφή διπλότυπων (τώρα που τα έχουμε όλα μαζί)
final_df.drop_duplicates(inplace=True)

print(f"Τελικό μέγεθος στο Pandas: {final_df.shape}")
print(f"Κατανάλωση μνήμης RAM: {final_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# 4. Αποθήκευση στο μαγικό μορφότυπο Parquet
output_file = 'cic_ids_2017_optimized.parquet'
final_df.to_parquet(output_file, engine='pyarrow', index=False)
print(f"\nΑποθηκεύτηκε επιτυχώς το: {output_file}")