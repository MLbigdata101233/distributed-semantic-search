#!/usr/bin/env python3
"""
Generate embeddings for the SuperUser data and combine with existing
AskUbuntu embeddings into a unified embeddings parquet.

Before running:
  1. Run preprocess_superuser.py first
  2. Make sure AskUbuntu embeddings still exist at EMBEDDINGS_PARQUET_DIR
"""
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config_gpu import *

# Paths
SUPERUSER_PARQUET = "/data/m25_jahanvi/ML_big/data/data1/pre_super"
EXISTING_EMBEDDINGS = "/data/m25_jahanvi/ML_big/GPU_project/data/data/embeddings_parquet_gpu"
COMBINED_OUTPUT = "/data/m25_jahanvi/ML_big/combined_data/data/combined_embeddings.parquet"

# ====== Load existing AskUbuntu embeddings ======
print("Loading existing AskUbuntu embeddings...")
existing_df = pd.read_parquet(EXISTING_EMBEDDINGS)
print(f"  AskUbuntu rows: {len(existing_df)}")
print(f"  Columns: {existing_df.columns.tolist()}")

# Tag existing data with source_site if not already
if 'source_site' not in existing_df.columns:
    existing_df['source_site'] = "askubuntu.com"
    print("  Added source_site='askubuntu.com' to existing data")

# ====== Load SuperUser preprocessed data ======
print(f"\nLoading SuperUser preprocessed data...")
new_df = pd.read_parquet(SUPERUSER_PARQUET)
print(f"  SuperUser rows: {len(new_df)}")

# texts = new_df['text_for_embedding'].tolist()
# ids = new_df['id'].tolist()
# source_sites = new_df['source_site'].tolist()

texts = new_df['text_for_embedding'].tolist()
ids = new_df['id'].tolist()

# Add source_site if not present (preprocessing didn't tag it)
if 'source_site' not in new_df.columns:
    new_df['source_site'] = "superuser.com"
    print("Added source_site='superuser.com' to SuperUser data")

source_sites = new_df['source_site'].tolist()

# Apply e5 passage prefix
texts_prefixed = [f"passage: {t}" for t in texts]

# ====== Generate SuperUser embeddings ======
print(f"\nLoading e5-large-v2 on GPU...")
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cuda")
model.eval()

batch_size = BATCH_SIZE_GPU
print(f"\nEmbedding {len(texts_prefixed)} SuperUser texts (batch_size={batch_size})...")

embeddings_list = []
for i in tqdm(range(0, len(texts_prefixed), batch_size), desc="Encoding"):
    batch = texts_prefixed[i:i+batch_size]
    with torch.no_grad():
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    embeddings_list.append(emb)

new_embeddings = np.vstack(embeddings_list).astype(np.float32)
print(f"\nSuperUser embeddings shape: {new_embeddings.shape}")

# ====== Build new SuperUser dataframe ======
new_emb_df = pd.DataFrame({
    "id": ids,
    "text": texts,                    # store original text without prefix
    "embedding": list(new_embeddings),
    "source_site": source_sites
})

# ====== Combine with existing ======
print(f"\nCombining datasets...")
print(f"  AskUbuntu: {len(existing_df)} rows")
print(f"  SuperUser: {len(new_emb_df)} rows")
combined_df = pd.concat([existing_df, new_emb_df], ignore_index=True)
print(f"  Combined:  {len(combined_df)} rows")

# Save
print(f"\nSaving to {COMBINED_OUTPUT}...")
combined_df.to_parquet(COMBINED_OUTPUT, index=False)
print("Done.")
print(f"\nNext step: rebuild FAISS index from {COMBINED_OUTPUT}")
