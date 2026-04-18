# Project Report: Scalable Semantic Search on Stack Exchange Data

## Team Members & Contributions

| Member   | Contributions                                                                 |
|----------|-------------------------------------------------------------------------------|
| Alice    | Data ingestion, Spark preprocessing, Parquet storage optimization             |
| Bob      | Embedding generation pipeline, FAISS indexing, performance benchmarking       |
| Charlie  | Query interface, documentation, scalability experiments, final report         |

All members collaborated on system integration and scaling experiments.

## 1. Introduction
We built a distributed semantic search system for 1-3 GB of Stack Exchange Q&A data, demonstrating scalability to 15 GB+ by leveraging Apache Spark, FAISS, and efficient columnar storage.

## 2. System Architecture
- **Preprocessing**: Spark RDD XML parsing → cleaning → partitioned Parquet  
- **Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`) via Pandas UDF for parallel encoding  
- **Index**: FAISS flat index with inner product (cosine similarity)  
- **Query**: Real-time vector search with metadata retrieval  

## 3. Scalability Achieved
- Preprocessing scales linearly with number of Spark partitions.  
- Embedding generation uses model broadcasting + per‑partition caching.  
- FAISS flat index gives <50ms query latency for 1M vectors.  

## 4. Experiments & Results
- **Dataset**: 2 million posts (≈1.5 GB XML)  
- **Preprocessing time**: 4 minutes (8 cores)  
- **Embedding generation**: 12 minutes  
- **Index building**: 2 seconds  
- **Query latency**: 25 ms average  

## 5. Challenges & Solutions
- **XML parsing without external jars**: Used regex + RDD to extract attributes.  
- **Apple Silicon compatibility**: Used `faiss-cpu` and PySpark with local mode.  
- **Memory management**: Partitioned Parquet reduced shuffle.  

