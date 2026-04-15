#!/usr/bin/env python3
"""
Command-line semantic search.
Example: python search.py "how to install python on ubuntu"
"""

import sys
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import *

# Load resources
print("Loading model...")
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)
print("Loading metadata...")
metadata = pd.read_parquet(METADATA_PATH)

def search(query, top_k=DEFAULT_TOP_K):
    # Encode and normalize query
    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        post_id = metadata.iloc[idx]["id"]
        results.append({"id": int(post_id), "score": float(dist)})
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search.py \"your search query\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    results = search(query)
    print(f"\nTop {len(results)} results for: {query}\n")
    for r in results:
        print(f"ID: {r['id']}   Score: {r['score']:.4f}")