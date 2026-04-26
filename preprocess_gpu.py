#!/usr/bin/env python3
"""
Spark-based preprocessing (CPU) – reads XML, cleans, saves as Parquet.
No cuDF, no RAPIDS – avoids binary incompatibility issues.
"""
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, concat_ws, when
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
import re
import html
from config_gpu import *

# Initialize Spark (local mode, adjust memory as needed)
import os
with open(".spark_cluster_ports") as f:
    cfg = dict(line.strip().split('=', 1) for line in f if '=' in line)

spark = SparkSession.builder \
    .appName("PreprocessOnCluster") \
    .master("spark://localhost:7077") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "3g") \
    .config("spark.executor.cores", "2") \
    .config("spark.cores.max", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# ---------- Clean HTML using a Pandas UDF (CPU) ----------
def clean_html(text):
    if text is None:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

clean_udf = udf(clean_html, StringType())

# ---------- Parse XML efficiently with Spark (no external jars) ----------
# Read XML as text file and filter row elements
raw_df = spark.read.text("/data/m25_jahanvi/ML_big/data/data1/Posts.xml")

# Filter lines containing <row
row_lines = raw_df.filter(col("value").contains("<row"))

# Parse attributes using a UDF (simple regex)
def parse_row(line):
    """Extract attributes from a <row ... /> line"""
    import re
    attrs = {}
    # Match key="value"
    pattern = re.compile(r'(\w+)="([^"]*)"')
    for k, v in pattern.findall(line):
        attrs[k] = v
    if "PostTypeId" not in attrs or "Body" not in attrs:
        return None
    return {
        "id": int(attrs.get("Id", 0)),
        "post_type_id": int(attrs["PostTypeId"]),
        "parent_id": int(attrs.get("ParentId", 0)) if attrs.get("ParentId") else None,
        "score": int(attrs.get("Score", 0)),
        "body_raw": attrs["Body"],
        "title": attrs.get("Title", ""),
        "tags": attrs.get("Tags", ""),
    }

parse_udf = udf(parse_row, StructType([
    StructField("id", IntegerType(), True),
    StructField("post_type_id", IntegerType(), True),
    StructField("parent_id", IntegerType(), True),
    StructField("score", IntegerType(), True),
    StructField("body_raw", StringType(), True),
    StructField("title", StringType(), True),
    StructField("tags", StringType(), True),
]))

parsed_df = row_lines.select(parse_udf("value").alias("row")).filter(col("row").isNotNull())
df = parsed_df.select("row.*")

# Clean and combine text
df = df.withColumn("body_clean", clean_udf(col("body_raw")))
df = df.withColumn("text_for_embedding",
    when(col("post_type_id") == 1,
         concat_ws(" ", col("title"), col("body_clean")))
    .otherwise(col("body_clean"))
).drop("body_raw")

# Save as Parquet (partitioned by year-month from creation_date if available, else not)
# For simplicity, we skip partitioning here
df.write.mode("overwrite").parquet("/data/m25_jahanvi/ML_big/data/data1/pre")

count = df.count()
print(f"Saved {count} rows to /data/m25_jahanvi/ML_big/data/data1/pre")
spark.stop()

