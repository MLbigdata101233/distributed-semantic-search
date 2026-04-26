#!/usr/bin/env python3
"""
Comprehensive benchmark for the semantic search system.

Produces:
  1. Query latency distribution (p50, p95, p99) - GPU and CPU
  2. Recall@10 vs ground truth (exact flat search)
  3. Recall vs nprobe tradeoff curve
  4. Index build time and disk size stats
  5. Throughput (queries per second)

Outputs:
  - benchmark_results.csv     : all numerical results
  - latency_distribution.png  : histogram of query latencies
  - recall_vs_nprobe.png      : recall@10 vs nprobe curve
  - latency_vs_nprobe.png     : latency vs nprobe (tradeoff)
  - summary.txt               : human-readable summary for the report
"""
import faiss
import numpy as np
import pandas as pd
import time
import os
import json
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from config_gpu import *

# -------------------- Setup --------------------
print("=" * 60)
print("SEMANTIC SEARCH SYSTEM BENCHMARK")
print("=" * 60)

# Load embeddings
print("\n[1/6] Loading embeddings and index...")
df = pd.read_parquet("/data/m25_jahanvi/ML_big/GPU_project/data/data/embeddings_parquet_gpu")
embeddings = np.vstack(df['embedding'].values).astype(np.float32)
n_vectors, dim = embeddings.shape
print(f"  Vectors: {n_vectors:,}  Dim: {dim}")

# Load IVF index from disk
cpu_index = faiss.read_index(FAISS_INDEX_PATH)
index_size_mb = os.path.getsize(FAISS_INDEX_PATH) / 1e6
print(f"  Index size on disk: {index_size_mb:.1f} MB")

# Move IVF index to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, GPU_ID, cpu_index)

# -------------------- Ground truth (exact search) --------------------
print("\n[2/6] Building exact (flat) index for ground truth...")
flat_index = faiss.IndexFlatIP(dim)
flat_gpu = faiss.index_cpu_to_gpu(res, GPU_ID, flat_index)
flat_gpu.add(embeddings)
print(f"  Flat index ready ({flat_gpu.ntotal} vectors)")

# -------------------- Generate test queries --------------------
print("\n[3/6] Generating test queries...")
N_QUERIES = 1000
np.random.seed(42)
# Use random embeddings from the dataset as queries (realistic)
query_idx = np.random.choice(n_vectors, N_QUERIES, replace=False)
query_vectors = embeddings[query_idx].copy()
print(f"  {N_QUERIES} test queries prepared")

# Get ground truth from exact search
print("  Computing ground truth with flat index...")
_, gt_indices = flat_gpu.search(query_vectors, 10)

# -------------------- Latency benchmark --------------------
print("\n[4/6] Measuring query latency (nprobe=32)...")
gpu_index.nprobe = 32

# Warm up
for _ in range(10):
    gpu_index.search(query_vectors[:1], 10)

# Single-query latency (realistic for a search API)
latencies_ms = []
for i in range(N_QUERIES):
    t0 = time.perf_counter()
    gpu_index.search(query_vectors[i:i+1], 10)
    t1 = time.perf_counter()
    latencies_ms.append((t1 - t0) * 1000)

latencies_ms = np.array(latencies_ms)
p50 = np.percentile(latencies_ms, 50)
p95 = np.percentile(latencies_ms, 95)
p99 = np.percentile(latencies_ms, 99)
mean_lat = np.mean(latencies_ms)

# Batch throughput
t0 = time.perf_counter()
gpu_index.search(query_vectors, 10)
t_batch = time.perf_counter() - t0
qps = N_QUERIES / t_batch

print(f"  Latency p50: {p50:.2f} ms")
print(f"  Latency p95: {p95:.2f} ms")
print(f"  Latency p99: {p99:.2f} ms")
print(f"  Mean: {mean_lat:.2f} ms")
print(f"  Batch throughput: {qps:.0f} queries/sec")

# -------------------- Recall vs nprobe --------------------
print("\n[5/6] Measuring recall@10 vs nprobe tradeoff...")
nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
recall_results = []

for np_val in nprobe_values:
    gpu_index.nprobe = np_val

    # Recall
    _, ann_indices = gpu_index.search(query_vectors, 10)
    recall_at_10 = np.mean([
        len(set(ann_indices[i]) & set(gt_indices[i])) / 10.0
        for i in range(N_QUERIES)
    ])

    # Latency at this nprobe
    sample_lat = []
    for _ in range(50):
        t0 = time.perf_counter()
        gpu_index.search(query_vectors[:1], 10)
        sample_lat.append((time.perf_counter() - t0) * 1000)
    avg_lat = np.mean(sample_lat)

    recall_results.append({
        'nprobe': np_val,
        'recall_at_10': recall_at_10,
        'latency_ms': avg_lat
    })
    print(f"  nprobe={np_val:4d}  recall@10={recall_at_10:.4f}  latency={avg_lat:.2f}ms")

recall_df = pd.DataFrame(recall_results)

# -------------------- Save results --------------------
print("\n[6/6] Saving results and generating plots...")
os.makedirs("benchmark_output", exist_ok=True)

# Combined CSV
recall_df.to_csv("benchmark_output/recall_vs_nprobe.csv", index=False)
pd.DataFrame({'latency_ms': latencies_ms}).to_csv(
    "benchmark_output/query_latencies.csv", index=False
)

# Plot 1: Latency distribution
plt.figure(figsize=(8, 5))
plt.hist(latencies_ms, bins=50, edgecolor='black', alpha=0.75)
plt.axvline(p50, color='green', linestyle='--', label=f'p50={p50:.1f}ms')
plt.axvline(p95, color='orange', linestyle='--', label=f'p95={p95:.1f}ms')
plt.axvline(p99, color='red', linestyle='--', label=f'p99={p99:.1f}ms')
plt.xlabel('Latency (ms)')
plt.ylabel('Number of queries')
plt.title(f'Query Latency Distribution (n={N_QUERIES}, nprobe=32)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('benchmark_output/latency_distribution.png', dpi=150)
plt.close()

# Plot 2: Recall vs nprobe
plt.figure(figsize=(8, 5))
plt.plot(recall_df['nprobe'], recall_df['recall_at_10'], 'o-', linewidth=2, markersize=8)
plt.xscale('log', base=2)
plt.xlabel('nprobe (clusters searched)')
plt.ylabel('Recall@10')
plt.title('Recall@10 vs nprobe (vs exact flat search)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('benchmark_output/recall_vs_nprobe.png', dpi=150)
plt.close()

# Plot 3: Latency vs nprobe (the tradeoff)
fig, ax1 = plt.subplots(figsize=(8, 5))
color1 = 'tab:blue'
ax1.set_xlabel('nprobe')
ax1.set_ylabel('Recall@10', color=color1)
ax1.plot(recall_df['nprobe'], recall_df['recall_at_10'], 'o-', color=color1, linewidth=2)
ax1.set_xscale('log', base=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Latency (ms)', color=color2)
ax2.plot(recall_df['nprobe'], recall_df['latency_ms'], 's--', color=color2, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Recall vs Latency Tradeoff')
plt.tight_layout()
plt.savefig('benchmark_output/recall_latency_tradeoff.png', dpi=150)
plt.close()

# Summary text
summary = f"""
SEMANTIC SEARCH BENCHMARK RESULTS
==================================

Dataset:
  Total vectors:    {n_vectors:,}
  Dimensions:       {dim}
  Embedding model:  {SENTENCE_TRANSFORMER_MODEL}
  Memory footprint: {embeddings.nbytes / 1e9:.2f} GB

Index:
  Type:             IVF1024,Flat
  Metric:           Inner Product (cosine on normalized vectors)
  Disk size:        {index_size_mb:.1f} MB
  GPU:              CUDA device {GPU_ID}

Query Latency (nprobe=32, single queries):
  p50:              {p50:.2f} ms
  p95:              {p95:.2f} ms
  p99:              {p99:.2f} ms
  Mean:             {mean_lat:.2f} ms
  Batch throughput: {qps:.0f} queries/sec

Recall@10 vs nprobe:
{recall_df.to_string(index=False)}

Best operating point (recall >= 0.95):
"""
best = recall_df[recall_df['recall_at_10'] >= 0.95]
if len(best) > 0:
    best_row = best.iloc[0]
    summary += f"  nprobe={int(best_row['nprobe'])}, recall={best_row['recall_at_10']:.4f}, latency={best_row['latency_ms']:.2f}ms\n"
else:
    summary += "  No nprobe achieves recall >= 0.95 (consider larger nprobe or IVF rebuild)\n"

with open('benchmark_output/summary.txt', 'w') as f:
    f.write(summary)

print(summary)
print("=" * 60)
print("All outputs saved to benchmark_output/")
print("=" * 60)