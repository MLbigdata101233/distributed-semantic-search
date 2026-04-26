#!/usr/bin/env python3
"""
Build FAISS index using CPU only.
At 945K vectors, CPU FAISS is plenty fast (~2-3 min build, ~5-10ms search)
and avoids the GPU SWIG binding issues with faiss-gpu wheels.

Use this if build_index_gpu.py fails with 'input not a numpy array'.
"""
import faiss
import numpy as np
import pandas as pd
import time
import os
from config_gpu import *

# Load embeddings
print("Loading embeddings...")
df = pd.read_parquet("/data/m25_jahanvi/ML_big/GPU_project/data/data/embeddings_parquet_gpu")

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
print(f"Loaded {n} vectors, dim={dim}")

# Build IVF-Flat on CPU
nlist = 1024
print(f"Creating IVF{nlist},Flat index (CPU)...")

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

t0 = time.time()
print("Training index...")
index.train(embeddings)
train_time = time.time() - t0

t0 = time.time()
print("Adding vectors...")
index.add(embeddings)
add_time = time.time() - t0

# Set nprobe for inference
index.nprobe = 32

print(f"\nIndex stats:")
print(f"  Train time: {train_time:.2f}s")
print(f"  Add time:   {add_time:.2f}s")
print(f"  Total vectors: {index.ntotal}")
print(f"  nprobe: {index.nprobe}")

# Save
print("Saving index...")
faiss.write_index(index, FAISS_INDEX_PATH)

# Save metadata WITH text
if 'text' in df.columns:
    metadata_df = pd.DataFrame({"id": ids, "text": df['text'].values})
else:
    print("WARNING: no 'text' column. Flask app won't show text.")
    metadata_df = pd.DataFrame({"id": ids})
metadata_df.to_parquet(METADATA_PATH)

print(f"Saved index to {FAISS_INDEX_PATH}")
print(f"Saved metadata to {METADATA_PATH}")

size_mb = os.path.getsize(FAISS_INDEX_PATH) / 1e6
print(f"Index file size: {size_mb:.1f} MB")
print("\nDone. CPU index ready - works in benchmark, Flask, and Spark scripts.")