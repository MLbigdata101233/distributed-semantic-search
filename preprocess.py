#!/usr/bin/env python3
"""
Distributed XML parsing with Apache Spark.
Output: Partitioned Parquet with cleaned text and metadata.
"""

import re
import html
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, trim, lower, concat_ws
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from config import *

# Initialize Spark
spark = SparkSession.builder \
    .appName(SPARK_APP_NAME) \
    .master(SPARK_MASTER) \
    .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

# ---------- Helper functions for cleaning ----------
def clean_html(text):
    """Remove HTML tags and unescape entities."""
    if text is None:
        return ""
    # Remove tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Unescape HTML entities
    text = html.unescape(text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

clean_html_udf = udf(clean_html, StringType())

# ---------- Parse XML using RDD (no external jars) ----------
def parse_row_element(line):
    """
    Parse a single XML row like:
    <row Id="1" PostTypeId="1" ... />
    Returns dict or None if not a row.
    """
    line = line.strip()
    if not line.startswith("<row"):
        return None
    # Simple regex to extract attributes (works for well-formed rows)
    pattern = re.compile(r'(\w+)="([^"]*)"')
    attrs = {}
    for key, value in pattern.findall(line):
        attrs[key] = value
    # Only keep posts with body text (questions and answers)
    if "PostTypeId" not in attrs or "Body" not in attrs:
        return None
    # Convert types
    record = {
        "id": int(attrs.get("Id", 0)),
        "post_type_id": int(attrs["PostTypeId"]),
        "parent_id": int(attrs.get("ParentId", 0)) if attrs.get("ParentId") else None,
        "creation_date": attrs.get("CreationDate"),
        "score": int(attrs.get("Score", 0)),
        "body_raw": attrs["Body"],
        "title": attrs.get("Title", ""),
        "tags": attrs.get("Tags", ""),
    }
    return record

# Read as text file and parse rows
raw_rdd = spark.sparkContext.textFile(RAW_XML_PATH)
row_rdd = raw_rdd.map(parse_row_element).filter(lambda x: x is not None)

# Convert to DataFrame
schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("post_type_id", IntegerType(), False),
    StructField("parent_id", IntegerType(), True),
    StructField("creation_date", StringType(), True),
    StructField("score", IntegerType(), True),
    StructField("body_raw", StringType(), True),
    StructField("title", StringType(), True),
    StructField("tags", StringType(), True),
])
df = spark.createDataFrame(row_rdd, schema=schema)

# Clean HTML and create combined text for embedding
df_clean = df.withColumn("body_clean", clean_html_udf(col("body_raw"))) \
             .withColumn("text_for_embedding",
                         when(col("post_type_id") == 1,
                              concat_ws(" ", col("title"), col("body_clean")))
                         .otherwise(col("body_clean")))

# Partition by creation date (year-month) if available, else simple partition
from pyspark.sql.functions import to_timestamp, date_format
# Parse ISO-like datetime; quote the literal 'T'. Also try a fallback without milliseconds.
df_clean = df_clean.withColumn("date", to_timestamp("creation_date", "yyyy-MM-dd'T'HH:mm:ss.SSS"))
df_clean = df_clean.withColumn(
    "date",
    when(col("date").isNull(), to_timestamp("creation_date", "yyyy-MM-dd'T'HH:mm:ss")).otherwise(col("date"))
)
df_clean = df_clean.withColumn("year_month", date_format("date", "yyyy-MM"))


# Write partitioned Parquet (one partition per month, improves later filtering)
# df_clean.write \
#     .mode("overwrite") \
#     .partitionBy("year_month") \
#     .parquet(PROCESSED_PARQUET_DIR)

# print(f"Preprocessing complete. Data saved to {PROCESSED_PARQUET_DIR}")
# spark.stop()

# --- Diagnostics + safer write with fallback ---
import sys, traceback, os
print("PROCESSED_PARQUET_DIR:", PROCESSED_PARQUET_DIR)
print("Output dir exists:", os.path.exists(PROCESSED_PARQUET_DIR))
print("DataFrame schema:")
df_clean.printSchema()
print("Sample year_month counts:")
try:
    df_clean.select("year_month").groupBy("year_month").count().show(10)
except Exception:
    pass

try:
    df_clean.write.mode("overwrite").partitionBy("year_month").parquet(PROCESSED_PARQUET_DIR)
except Exception as e:
    print("Parquet write failed. Exception:")
    traceback.print_exc()
    print("Attempting a small write to /tmp/test_parquet to rule out permissions/path issues...")
    try:
        df_clean.limit(10).write.mode("overwrite").parquet("/tmp/test_parquet")
        print("Small write to /tmp/test_parquet succeeded. Check permissions for target dir.")
    except Exception:
        print("Small write to /tmp also failed. See traceback above.")
        traceback.print_exc()
    sys.exit(1)

print(f"Preprocessing complete. Data saved to {PROCESSED_PARQUET_DIR}")
spark.stop()