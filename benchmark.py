#!/usr/bin/env python3
"""
Benchmark pipeline components on increasing dataset sizes.
Assumes raw XML is already available; subsamples rows.
"""

import time
import subprocess
import pandas as pd
import numpy as np
from config import *

def run_command(cmd):
    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    return time.time() - start

def benchmark():
    scales = [0.25, 0.5, 1.0]   # fraction of original dataset (relative to 2M rows)
    results = []
    for frac in scales:
        print(f"\n=== Benchmark at {frac*100}% scale ===")
        # Create subset XML (simulate by copying first N rows)
        # For real benchmarking, you would adjust download_dataset.py to extract different sizes.
        # Here we assume we have Posts_subset.xml and we create a smaller copy.
        subset_file = f"data/raw/Posts_{int(frac*100)}pct.xml"
        # Simple head -n (approx 1.5M rows per GB, but we just count rows)
        # Not perfect but demonstrates scaling.
        cmd = f"head -n {int(frac*2000000)} data/raw/Posts_subset.xml > {subset_file}"
        subprocess.run(cmd, shell=True)
        
        # Update config temporarily
        RAW_XML_PATH = subset_file
        
        # Time preprocessing
        t_pre = run_command("python preprocess.py")
        # Time embedding generation
        t_emb = run_command("python generate_embeddings.py")
        # Time indexing
        t_idx = run_command("python build_index.py")
        # Time 10 queries
        query_times = []
        for q in ["python error", "ubuntu update", "disk full"]:
            start = time.time()
            subprocess.run(f"python search.py \"{q}\" > /dev/null", shell=True)
            query_times.append(time.time() - start)
        avg_query = np.mean(query_times)
        
        results.append({
            "scale": frac,
            "preprocess_sec": t_pre,
            "embed_sec": t_emb,
            "index_sec": t_idx,
            "avg_query_ms": avg_query * 1000
        })
    
    df = pd.DataFrame(results)
    print("\n=== Benchmark Results ===")
    print(df)
    df.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    benchmark()