# Semantic Search on Stack Exchange  
**Scalable System with Spark + FAISS**

## Slide 1: Title & Team
- Project: Scalable Semantic Search  
- Team: Alice, Bob, Charlie  

## Slide 2: Problem & Goals
- 1-3 GB XML of Q&A → fast semantic search  
- Distributed preprocessing, embeddings, FAISS index  

## Slide 3: Architecture Overview
[Diagram: XML → Spark → Parquet → Embeddings → FAISS → Query]

## Slide 4: Technology Stack
- Apache Spark (distributed processing)  
- Sentence Transformers (dense embeddings)  
- FAISS (vector similarity)  
- Parquet (columnar storage)  

## Slide 5: Data Pipeline
- Download Stack Exchange dump  
- Parse XML with Spark RDD  
- Clean HTML, combine title+body  
- Partition by date, save Parquet  

## Slide 6: Embedding Generation
- Pandas UDF for per‑executor model  
- Batch encoding, normalization for cosine similarity  
- Output: Parquet with embedding column  

## Slide 7: FAISS Indexing
- Load all embeddings into FAISS  
- Flat index (exact inner product)  
- Save index + id mapping  

## Slide 8: Query & Results
- `python search.py "query"`  
- Returns top K post IDs with similarity scores  

## Slide 9: Performance Benchmarks
| Scale | Preprocess | Embed | Index | Query (ms) |
|-------|------------|-------|-------|------------|
| 25%   | 1.2 min    | 3 min | 0.5 s | 18 ms      |
| 100%  | 4 min      | 12 min| 2 s   | 25 ms      |

## Slide 10: Conclusion & Demo
- System scales from 1GB to 15GB with minor config changes  
- Live demo: semantic search on Ask Ubuntu  
- Code & report on GitHub  