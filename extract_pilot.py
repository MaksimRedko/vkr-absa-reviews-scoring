import sqlite3
import csv
import os

db_path = 'data/dataset.db'
output_path = 'outputs/golden_pilot_raw.csv'

os.makedirs('outputs', exist_ok=True)

query = """
SELECT id, nm_id, rating, full_text, true_labels
FROM reviews
WHERE true_labels IS NOT NULL 
  AND true_labels != ''
  AND length(full_text) > 30
ORDER BY RANDOM()
LIMIT 50;
"""

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    colnames = [description[0] for description in cursor.description]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(colnames)
        writer.writerows(rows)
    
    print(f"Successfully exported {len(rows)} rows to {output_path}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
