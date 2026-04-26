#!/usr/bin/env python3
"""CPU-only benchmark."""
import faiss
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from config_gpu import *

print("=" * 60)
print("SEMANTIC SEARCH BENCHMARK (CPU)")
print("=" * 60)

print("\n[1/6] Loading embeddings and index...")
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

n_vectors, dim = embeddings.shape
print(f"  Vectors: {n_vectors:,}  Dim: {dim}")

ivf_index = faiss.read_index(FAISS_INDEX_PATH)
index_size_mb = os.path.getsize(FAISS_INDEX_PATH) / 1e6
print(f"  Index size on disk: {index_size_mb:.1f} MB")

print("\n[2/6] Building exact (flat) CPU index for ground truth...")
flat_index = faiss.IndexFlatIP(dim)
flat_index.add(embeddings)
print(f"  Flat index ready: {flat_index.ntotal} vectors")

print("\n[3/6] Generating test queries...")
N_QUERIES = 1000
np.random.seed(42)
query_idx = np.random.choice(n_vectors, N_QUERIES, replace=False)
query_vectors = np.ascontiguousarray(embeddings[query_idx], dtype=np.float32)

print("  Computing ground truth (CPU brute-force)...")
t0 = time.time()
_, gt_indices = flat_index.search(query_vectors, 10)
print(f"  Ground truth done in {time.time()-t0:.1f}s")

print("\n[4/6] Measuring latency (nprobe=32)...")
ivf_index.nprobe = 32
for _ in range(10):
    ivf_index.search(query_vectors[:1], 10)

latencies_ms = []
for i in range(N_QUERIES):
    t0 = time.perf_counter()
    ivf_index.search(query_vectors[i:i+1], 10)
    latencies_ms.append((time.perf_counter() - t0) * 1000)

latencies_ms = np.array(latencies_ms)
p50 = np.percentile(latencies_ms, 50)
p95 = np.percentile(latencies_ms, 95)
p99 = np.percentile(latencies_ms, 99)
mean_lat = np.mean(latencies_ms)

t0 = time.perf_counter()
ivf_index.search(query_vectors, 10)
qps = N_QUERIES / (time.perf_counter() - t0)

print(f"  p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms")
print(f"  mean={mean_lat:.2f}ms throughput={qps:.0f} qps")

print("\n[5/6] Recall@10 vs nprobe...")
nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
recall_results = []
for np_val in nprobe_values:
    ivf_index.nprobe = np_val
    _, ann_indices = ivf_index.search(query_vectors, 10)
    recall = np.mean([
        len(set(ann_indices[i]) & set(gt_indices[i])) / 10.0
        for i in range(N_QUERIES)
    ])
    sample_lat = []
    for _ in range(50):
        t0 = time.perf_counter()
        ivf_index.search(query_vectors[:1], 10)
        sample_lat.append((time.perf_counter() - t0) * 1000)
    avg_lat = np.mean(sample_lat)
    recall_results.append({'nprobe': np_val, 'recall_at_10': recall, 'latency_ms': avg_lat})
    print(f"  nprobe={np_val:4d}  recall={recall:.4f}  lat={avg_lat:.2f}ms")

recall_df = pd.DataFrame(recall_results)

print("\n[6/6] Saving outputs...")
os.makedirs("benchmark_output", exist_ok=True)
recall_df.to_csv("benchmark_output/recall_vs_nprobe.csv", index=False)
pd.DataFrame({'latency_ms': latencies_ms}).to_csv("benchmark_output/query_latencies.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(latencies_ms, bins=50, edgecolor='black', alpha=0.75)
plt.axvline(p50, color='green', linestyle='--', label=f'p50={p50:.1f}ms')
plt.axvline(p95, color='orange', linestyle='--', label=f'p95={p95:.1f}ms')
plt.axvline(p99, color='red', linestyle='--', label=f'p99={p99:.1f}ms')
plt.xlabel('Latency (ms)')
plt.ylabel('Number of queries')
plt.title(f'Query Latency Distribution (n={N_QUERIES})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('benchmark_output/latency_distribution.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(recall_df['nprobe'], recall_df['recall_at_10'], 'o-', linewidth=2, markersize=8)
plt.xscale('log', base=2)
plt.xlabel('nprobe')
plt.ylabel('Recall@10')
plt.title('Recall@10 vs nprobe')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('benchmark_output/recall_vs_nprobe.png', dpi=150)
plt.close()

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlabel('nprobe')
ax1.set_ylabel('Recall@10', color='tab:blue')
ax1.plot(recall_df['nprobe'], recall_df['recall_at_10'], 'o-', color='tab:blue', linewidth=2)
ax1.set_xscale('log', base=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.set_ylabel('Latency (ms)', color='tab:red')
ax2.plot(recall_df['nprobe'], recall_df['latency_ms'], 's--', color='tab:red', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:red')
plt.title('Recall vs Latency Tradeoff')
plt.tight_layout()
plt.savefig('benchmark_output/recall_latency_tradeoff.png', dpi=150)
plt.close()

summary = f"""
SEMANTIC SEARCH BENCHMARK RESULTS
=================================

Dataset:
  Total vectors:    {n_vectors:,}
  Dimensions:       {dim}
  Embedding model:  {SENTENCE_TRANSFORMER_MODEL}
  Memory:           {embeddings.nbytes / 1e9:.2f} GB

Index:
  Type:             IVF1024,Flat
  Disk size:        {index_size_mb:.1f} MB

Query Latency (nprobe=32):
  p50:              {p50:.2f} ms
  p95:              {p95:.2f} ms
  p99:              {p99:.2f} ms
  Mean:             {mean_lat:.2f} ms
  Throughput:       {qps:.0f} qps

Recall@10 vs nprobe:
{recall_df.to_string(index=False)}
"""
best = recall_df[recall_df['recall_at_10'] >= 0.95]
if len(best) > 0:
    b = best.iloc[0]
    summary += f"\nBest (recall>=0.95): nprobe={int(b['nprobe'])}, recall={b['recall_at_10']:.4f}, lat={b['latency_ms']:.2f}ms\n"

with open('benchmark_output/summary.txt', 'w') as f:
    f.write(summary)

print(summary)
print("=" * 60)
print("All outputs in benchmark_output/")
print("=" * 60)
