import pandas as pd
import numpy as np
from compute_all_metrics import compute_all_metrics
from clean_and_split import prepare_text_from_string as prepare_text
from global_model import get_vector
from global_model import load_model
from tqdm import tqdm  # progress bar

# === Load embedding model ===
load_model("C:/Users/alons/OneDrive/שולחן העבודה/Text_Complexity/Models/word2vecf.npz")


# === Feature extraction function ===
def extract_features(text):
    sentences = prepare_text(text)
    tokens = [word for sent in sentences for word in sent]
    embeddings = [vec for w in tokens if (vec := get_vector(w)) is not None]
    if len(embeddings) < 5:
        return None
    return compute_all_metrics(np.array(embeddings))


# === Load data ===
df = pd.read_excel("clear_corpus.xlsx")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces in column names

feature_rows = []

# === Loop through the whole file with progress bar ===
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
    try:
        metrics = extract_features(row['Excerpt'])
        if metrics is not None:
            metrics['BT_score'] = row['BT_easiness']
            metrics['SE'] = row['s.e.']
            metrics['id'] = row['ID'] if 'ID' in row else idx
            feature_rows.append(metrics)
    except Exception as e:
        print(f"Skipping row {idx} due to error: {e}")

# === Save to CSV ===
data = pd.DataFrame(feature_rows).dropna()
data.to_csv("clear_features.csv", index=False)

print("Feature extraction complete. Saved to 'clear_features.csv'")
