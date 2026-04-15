#!/usr/bin/env python3
"""
Build FAISS index from embedding Parquet.
Also saves mapping from index position to post ID.
"""

import faiss
import numpy as np
import pandas as pd
from config import *

print("Loading embeddings from", EMBEDDINGS_PARQUET_DIR)
df = pd.read_parquet(EMBEDDINGS_PARQUET_DIR)
# Extract embeddings (list of lists -> numpy float32)
embeddings = np.vstack(df["embedding"].values).astype(np.float32)
ids = df["id"].values

dim = embeddings.shape[1]
print(f"Total vectors: {len(embeddings)}, dimension: {dim}")

# Choose index type
if FAISS_INDEX_TYPE == "Flat":
    index = faiss.IndexFlatIP(dim)   # Inner product (cosine on normalized vectors)
elif FAISS_INDEX_TYPE.startswith("IVF"):
    # Example: "IVF100,PQ16" – requires training
    raise NotImplementedError("IVF index requires training; use Flat for simplicity")
else:
    index = faiss.IndexFlatL2(dim)

print("Adding vectors to index...")
index.add(embeddings)

print(f"Saving index to {FAISS_INDEX_PATH}")
faiss.write_index(index, FAISS_INDEX_PATH)

# Save metadata mapping
metadata_df = pd.DataFrame({"id": ids})
metadata_df.to_parquet(METADATA_PATH)
print(f"Metadata saved to {METADATA_PATH}")