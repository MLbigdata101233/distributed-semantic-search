#!/usr/bin/env python3
"""
Rebuild FAISS index from the combined AskUbuntu + SuperUser embeddings.
The metadata parquet now includes a source_site column so the Flask app
can construct the correct URL for each result.
"""
import faiss
import numpy as np
import pandas as pd
import time
import os
from config_gpu import *

COMBINED_EMBEDDINGS = "/data/m25_jahanvi/ML_big/combined_data/data/combined_embeddings_parquet"

print(f"Loading combined embeddings from {COMBINED_EMBEDDINGS}...")
df = pd.read_parquet(COMBINED_EMBEDDINGS)
print(f"  Total rows: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Source distribution:")
print(df['source_site'].value_counts())

embeddings_raw = df['embedding'].values
if embeddings_raw.dtype == object:
    embeddings = np.array(
        [np.asarray(e, dtype=np.float32) for e in embeddings_raw],
        dtype=np.float32
    )
else:
    embeddings = np.vstack(embeddings_raw).astype(np.float32)
embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

ids = df['id'].values
dim = embeddings.shape[1]
n = len(embeddings)
print(f"\n  Embeddings: {embeddings.shape}, dtype={embeddings.dtype}")
print(f"  Memory: {embeddings.nbytes / 1e9:.2f} GB")

# Build IVF-Flat index. nlist scales with N.
nlist = 1024
print(f"\nBuilding IVF{nlist},Flat index (CPU)...")

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

t0 = time.time()
print("Training index...")
index.train(embeddings)
print(f"  Train time: {time.time()-t0:.1f}s")

t0 = time.time()
print("Adding vectors...")
index.add(embeddings)
print(f"  Add time: {time.time()-t0:.1f}s")

index.nprobe = 32

print(f"\nIndex stats:")
print(f"  Total vectors: {index.ntotal}")
print(f"  nprobe: {index.nprobe}")

# Save index (overwrites the old single-source one)
print(f"\nSaving index to {FAISS_INDEX_PATH}...")
faiss.write_index(index, FAISS_INDEX_PATH)

# Save metadata WITH source_site column
print(f"Saving metadata with source_site...")
metadata_df = pd.DataFrame({
    "id": ids,
    "text": df['text'].values,
    "source_site": df['source_site'].values
})
metadata_df.to_parquet(METADATA_PATH)

print(f"\nDone.")
print(f"  Index size: {os.path.getsize(FAISS_INDEX_PATH)/1e6:.1f} MB")
print(f"  Metadata rows: {len(metadata_df)}")
print(f"\nNow update query_app.py to use per-row source_site for URLs.")
