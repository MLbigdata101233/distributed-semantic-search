#!/usr/bin/env python3
"""
Batch semantic search using Spark for query distribution.

This is a legitimate use of Spark + FAISS: instead of trying to distribute
a single query (where Spark's overhead exceeds FAISS's latency), we use
Spark to distribute MANY queries across workers in parallel.

Each worker loads the FAISS index + embedding model in its partition,
processes its share of queries, and emits results. Spark handles the
distribution, fault tolerance, and aggregation.

Usage:
    python batch_search_spark.py queries.txt           # one query per line
    python batch_search_spark.py --evaluate            # run recall evaluation
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, ArrayType
)
from config_gpu import *


def read_master_url():
    """Read the cluster master URL from .spark_cluster_ports if it exists,
    otherwise fall back to local mode."""
    if os.path.exists(".spark_cluster_ports"):
        with open(".spark_cluster_ports") as f:
            for line in f:
                if line.startswith("SPARK_MASTER_URL="):
                    return line.strip().split("=", 1)[1]
    return "local[*]"


def search_partition(queries_iter):
    """
    Worker-side function: loads model + index ONCE per partition,
    then processes all queries in that partition.

    This is the key pattern - heavy resources loaded per partition,
    not per query.
    """
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # Heavy initialization - happens once per partition
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cpu")
    # Note: CPU here because workers don't have exclusive GPU access.
    # In a real multi-GPU cluster, you'd assign GPUs per executor.
    index = faiss.read_index(FAISS_INDEX_PATH)
    index.nprobe = 32

    metadata = pd.read_parquet(METADATA_PATH)

    # Process each query in this partition
    results = []
    for query_row in queries_iter:
        query_id = query_row.query_id
        query_text = query_row.query_text

        # Encode with e5 query prefix
        prefixed = f"query: {query_text}"
        vec = model.encode([prefixed], normalize_embeddings=True).astype(np.float32)
        distances, indices = index.search(vec, 10)

        # Format results
        result_ids = []
        result_scores = []
        for d, i in zip(distances[0], indices[0]):
            if i != -1:
                result_ids.append(int(metadata.iloc[i]['id']))
                result_scores.append(float(d))

        results.append((query_id, query_text, result_ids, result_scores))

    return iter(results)


def run_batch_search(spark, queries):
    """
    Distribute queries across Spark workers, search in parallel,
    return aggregated results.
    """
    # Build a DataFrame of queries with IDs
    query_data = [(i, q) for i, q in enumerate(queries)]
    schema = StructType([
        StructField("query_id", IntegerType()),
        StructField("query_text", StringType())
    ])
    df = spark.createDataFrame(query_data, schema)

    # Repartition to distribute work across workers
    n_partitions = max(2, min(len(queries) // 10, 8))
    df = df.repartition(n_partitions)
    print(f"Distributing {len(queries)} queries across {n_partitions} partitions")

    # Run the distributed search
    result_schema = StructType([
        StructField("query_id", IntegerType()),
        StructField("query_text", StringType()),
        StructField("result_ids", ArrayType(IntegerType())),
        StructField("result_scores", ArrayType(FloatType()))
    ])

    t_start = time.time()
    results_rdd = df.rdd.mapPartitions(search_partition)
    results_df = spark.createDataFrame(results_rdd, result_schema)
    results = results_df.collect()
    elapsed = time.time() - t_start

    return results, elapsed


def cmd_search(args):
    """Run batch search on a file of queries."""
    if not os.path.exists(args.queries_file):
        print(f"ERROR: queries file not found: {args.queries_file}")
        sys.exit(1)

    with open(args.queries_file) as f:
        queries = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(queries)} queries from {args.queries_file}")

    master_url = read_master_url()
    print(f"Connecting to: {master_url}")

    spark = (SparkSession.builder
             .appName("BatchSemanticSearch")
             .master(master_url)
             .config("spark.executor.memory", "6g")
             .config("spark.driver.memory", "8g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    results, elapsed = run_batch_search(spark, queries)

    print(f"\nCompleted {len(results)} searches in {elapsed:.1f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} queries/sec\n")

    # Print results
    for r in results[:10]:  # show first 10
        print(f"Query: {r.query_text}")
        for rid, score in zip(r.result_ids[:3], r.result_scores[:3]):
            print(f"  ID={rid} score={score:.4f}")
        print()

    # Save full results
    output_path = "batch_search_results.csv"
    pd.DataFrame([{
        "query_id": r.query_id,
        "query_text": r.query_text,
        "top_ids": str(r.result_ids),
        "top_scores": str([f"{s:.4f}" for s in r.result_scores])
    } for r in results]).to_csv(output_path, index=False)
    print(f"Full results saved to {output_path}")

    spark.stop()


def cmd_evaluate(args):
    """Run a small built-in evaluation set."""
    sample_queries = [
        "how to install python on ubuntu",
        "git merge conflict resolution",
        "tcp three way handshake explanation",
        "python list comprehension examples",
        "react useeffect hook tutorial",
        "linux find command syntax",
        "javascript promise async await",
        "docker compose multi container",
        "sql join types difference",
        "vim exit without saving",
        "regex match email address",
        "nginx reverse proxy configuration",
        "kubernetes pod vs deployment",
        "ssh key authentication setup",
        "python virtual environment venv",
        "css flexbox vs grid layout",
        "java garbage collection tuning",
        "mongodb vs postgres differences",
        "bash script for loop syntax",
        "rest api authentication methods",
    ]

    master_url = read_master_url()
    print(f"Connecting to: {master_url}")
    print(f"Running batch evaluation with {len(sample_queries)} sample queries")

    spark = (SparkSession.builder
             .appName("BatchSearchEvaluation")
             .master(master_url)
             .config("spark.executor.memory", "6g")
             .config("spark.driver.memory", "8g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    results, elapsed = run_batch_search(spark, sample_queries)

    # Statistics
    print(f"\n{'='*60}")
    print(f"BATCH SEARCH EVALUATION")
    print(f"{'='*60}")
    print(f"Total queries:      {len(results)}")
    print(f"Total elapsed:      {elapsed:.2f}s")
    print(f"Avg per query:      {elapsed/len(results)*1000:.1f}ms")
    print(f"Throughput:         {len(results)/elapsed:.1f} queries/sec")
    print(f"\nFirst 5 results:")
    for r in results[:5]:
        print(f"\n  Query: {r.query_text}")
        for rid, score in zip(r.result_ids[:3], r.result_scores[:3]):
            print(f"    ID={rid} score={score:.4f}")

    # Save
    pd.DataFrame([{
        "query_id": r.query_id,
        "query_text": r.query_text,
        "top_ids": str(r.result_ids),
        "top_scores": str([f"{s:.4f}" for s in r.result_scores])
    } for r in results]).to_csv("batch_search_evaluation.csv", index=False)
    print(f"\nResults saved to batch_search_evaluation.csv")
    print(f"{'='*60}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    p_search = sub.add_parser("search", help="Search a file of queries")
    p_search.add_argument("queries_file", help="File with one query per line")

    p_eval = sub.add_parser("evaluate", help="Run built-in evaluation queries")

    args = parser.parse_args()

    if args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)
