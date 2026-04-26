# Paths
RAW_XML_PATH = "data/data/raw/Posts.xml"   # full 200 GB
SUBSET_XML_PATH = "data/data/raw/Posts_20M.xml"   # we'll create subset
PROCESSED_PARQUET_DIR = "/data/m25_jahanvi/ML_big/data/data1"
EMBEDDINGS_PARQUET_DIR = "data/data/embeddings_parquet_gpu"
FAISS_INDEX_PATH = "data/data/faiss_gpu.index"
METADATA_PATH = "data/data/metadata_gpu.parquet"

# Spark GPU config
SPARK_MASTER = "spark://localhost:7077"   # standalone cluster
SPARK_DRIVER_MEMORY = "32g"
SPARK_EXECUTOR_MEMORY = "48g"              # A6000 has 48GB
SPARK_EXECUTOR_CORES = 8
SPARK_NUM_EXECUTORS = 2                    # can run multiple on same GPU? careful
# RAPIDS settings
SPARK_RAPIDS_CONF = {
    "spark.plugins": "com.nvidia.spark.SQLPlugin",
    "spark.rapids.sql.concurrentGpuTasks": "2",
    "spark.rapids.memory.pooling.enabled": "true",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
}

# Embedding model (larger, better quality)
SENTENCE_TRANSFORMER_MODEL = "intfloat/e5-large-v2"   # 1024 dim, excellent
BATCH_SIZE_GPU = 4096

# FAISS GPU index
FAISS_INDEX_TYPE = "IVF4096,PQ64"   # for 20M vectors
FAISS_METRIC = "inner_product"
GPU_ID = 0