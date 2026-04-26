#!/usr/bin/env python3
"""
Generate embeddings using e5-large-v2 with proper passage prefix.
e5 models REQUIRE "passage: " prefix on documents and "query: " on queries.
"""
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config_gpu import *

# Load processed Parquet
df = pd.read_parquet("/data/m25_jahanvi/ML_big/GPU_project/data/data/processed_parquet_gpu")
texts = df['text_for_embedding'].tolist()
ids = df['id'].tolist()

# CRITICAL FIX: e5 models require "passage: " prefix for documents
# Without this, retrieval quality drops measurably
texts_prefixed = [f"passage: {t}" for t in texts]

print(f"Encoding {len(texts_prefixed)} documents with e5 passage prefix...")

# Load model on GPU
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cuda")
model.eval()

# Batch encoding
batch_size = BATCH_SIZE_GPU
embeddings_list = []
for i in tqdm(range(0, len(texts_prefixed), batch_size), desc="Encoding"):
    batch = texts_prefixed[i:i+batch_size]
    with torch.no_grad():
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    embeddings_list.append(emb)

embeddings = np.vstack(embeddings_list).astype(np.float32)
print(f"Embeddings shape: {embeddings.shape}")

# Save embeddings + ids + original text (for the Flask demo later)
df_out = pd.DataFrame({
    "id": ids,
    "text": texts,  # save original text without prefix for display
    "embedding": list(embeddings)
})
df_out.to_parquet(EMBEDDINGS_PARQUET_DIR, index=False)
print(f"Saved embeddings to {EMBEDDINGS_PARQUET_DIR}")