#!/usr/bin/env python3
"""Diagnose what's in the embeddings parquet."""
import numpy as np
import pandas as pd
import faiss

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"faiss version: {faiss.__version__}")
print(f"faiss has GPU: {hasattr(faiss, 'StandardGpuResources')}")

print("\n" + "=" * 60)
print("LOADING PARQUET")
print("=" * 60)
df = pd.read_parquet("/data/m25_jahanvi/ML_big/GPU_project/data/data/embeddings_parquet_gpu")
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Column dtypes:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print("\n" + "=" * 60)
print("EMBEDDING COLUMN INSPECTION")
print("=" * 60)
emb_col = df['embedding']
print(f"Series dtype: {emb_col.dtype}")
print(f"First element type: {type(emb_col.iloc[0])}")
print(f"First element preview: {str(emb_col.iloc[0])[:120]}...")

first = emb_col.iloc[0]
if hasattr(first, '__len__'):
    print(f"First element length: {len(first)}")
if hasattr(first, 'dtype'):
    print(f"First element dtype: {first.dtype}")
if hasattr(first, 'shape'):
    print(f"First element shape: {first.shape}")

print("\n" + "=" * 60)
print("CONVERSION ATTEMPT")
print("=" * 60)
embeddings_raw = emb_col.values
print(f"Raw values dtype: {embeddings_raw.dtype}")
print(f"Raw values shape: {embeddings_raw.shape}")

# Try the row-by-row conversion
print("\nConverting row-by-row...")
try:
    embeddings = np.array(
        [np.asarray(e, dtype=np.float32) for e in embeddings_raw[:5]],
        dtype=np.float32
    )
    print(f"First 5 rows -> shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"Contiguous: {embeddings.flags['C_CONTIGUOUS']}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nFull conversion...")
embeddings = np.array(
    [np.asarray(e, dtype=np.float32) for e in embeddings_raw],
    dtype=np.float32
)
embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
print(f"Full shape: {embeddings.shape}")
print(f"Full dtype: {embeddings.dtype}")
print(f"Full contiguous: {embeddings.flags['C_CONTIGUOUS']}")
print(f"Memory layout: F={embeddings.flags['F_CONTIGUOUS']}, C={embeddings.flags['C_CONTIGUOUS']}")
print(f"Has NaN: {np.isnan(embeddings).any()}")
print(f"Has Inf: {np.isinf(embeddings).any()}")
print(f"Min: {embeddings.min()}, Max: {embeddings.max()}")

print("\n" + "=" * 60)
print("FAISS COMPATIBILITY TEST")
print("=" * 60)
# Try training a tiny CPU index first
print("Testing CPU IndexFlatIP.add() with first 1000 rows...")
try:
    test_index = faiss.IndexFlatIP(embeddings.shape[1])
    test_index.add(embeddings[:1000])
    print(f"SUCCESS: CPU index now has {test_index.ntotal} vectors")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\nTesting CPU IndexIVFFlat.train() with first 10000 rows...")
try:
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    test_index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 64, faiss.METRIC_INNER_PRODUCT)
    test_index.train(embeddings[:10000])
    print(f"SUCCESS: CPU index trained")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\nTesting GPU resources...")
try:
    res = faiss.StandardGpuResources()
    print("SUCCESS: StandardGpuResources created")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\nTesting GPU index_cpu_to_gpu...")
try:
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    cpu_index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 64, faiss.METRIC_INNER_PRODUCT)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    print("SUCCESS: index moved to GPU")
    print("Testing GPU train with first 10000 rows...")
    gpu_index.train(embeddings[:10000])
    print(f"SUCCESS: GPU train works")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
