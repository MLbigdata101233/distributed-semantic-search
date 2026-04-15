#!/usr/bin/env python3
"""
Parallel embedding generation using Pandas UDF.
Reads cleaned Parquet, adds embedding column, writes new Parquet.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import *

spark = SparkSession.builder \
    .appName(SPARK_APP_NAME + "_Embeddings") \
    .master(SPARK_MASTER) \
    .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
    .getOrCreate()

# Load processed data
df = spark.read.parquet(PROCESSED_PARQUET_DIR)

# ----- Pandas UDF for batched embeddings -----
model_name = SENTENCE_TRANSFORMER_MODEL

@pandas_udf(returnType=ArrayType(FloatType()))
def embed_text(text_series: pd.Series) -> pd.Series:
    """Generate embeddings for a batch of texts."""
    # Cache model per worker (singleton per process)
    if not hasattr(embed_text, "model"):
        embed_text.model = SentenceTransformer(model_name)
    model = embed_text.model
    # Encode with default settings (normalize for cosine similarity)
    embeddings = model.encode(text_series.tolist(), normalize_embeddings=True)
    return pd.Series(embeddings.tolist())

# Apply UDF (adds a new column of arrays)
df_emb = df.withColumn("embedding", embed_text(col("text_for_embedding")))

# Drop raw text fields to save space (keep only needed)
df_emb = df_emb.drop("body_raw", "body_clean", "text_for_embedding")

# Write as single Parquet (no partition, easier for FAISS)
df_emb.write.mode("overwrite").parquet(EMBEDDINGS_PARQUET_DIR)

print(f"Embeddings saved to {EMBEDDINGS_PARQUET_DIR}")
spark.stop()