import pandas as pd
import os
import json

csv_paths = [
    "benchmark/eval_datasets/combined_benchmark.csv",
    "parser/razmetka/checked_reviews.csv"
]

dfs = []
for p in csv_paths:
    if os.path.exists(p):
        try:
            df_item = pd.read_csv(p)
            # Ensure columns exist
            required = ['id', 'nm_id', 'rating', 'full_text', 'true_labels']
            if all(col in df_item.columns for col in required):
                dfs.append(df_item[required])
            else:
                print(f"Skipping {p}: missing columns. Found: {df_item.columns.tolist()}")
        except Exception as e:
            print(f"Error reading {p}: {e}")

if not dfs:
    print("No valid CSV data found with labels!")
    exit(1)

df = pd.concat(dfs).drop_duplicates(subset=['id'])

# Filter
# true_labels might be string or None
df = df[df['true_labels'].notna() & (df['true_labels'] != '') & (df['true_labels'] != '{}')]
df = df[df['full_text'].str.len() > 30]

# Sample 50
sample_size = min(50, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

output_path = 'outputs/golden_pilot_raw.csv'
os.makedirs('outputs', exist_ok=True)
df_sample.to_csv(output_path, index=False, encoding='utf-8')

print(f"Successfully exported {len(df_sample)} rows to {output_path}")
