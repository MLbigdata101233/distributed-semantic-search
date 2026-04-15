#!/bin/bash
set -e
echo "Step 1: Download dataset (if not already present)"
python download_dataset.py

echo "Step 2: Preprocess XML to Parquet"
python preprocess.py

echo "Step 3: Generate embeddings"
python generate_embeddings.py

echo "Step 4: Build FAISS index"
python build_index.py

echo "Pipeline complete. Test search:"
python search.py "apache spark configuration"