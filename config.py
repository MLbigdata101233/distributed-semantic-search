# Paths
RAW_XML_PATH = "data/data/raw/Posts.xml"
PROCESSED_PARQUET_DIR = "data/data/processed_parquet"
EMBEDDINGS_PARQUET_DIR = "data/data/embeddings_parquet"
FAISS_INDEX_PATH = "data/data/faiss_index.bin"
METADATA_PATH = "data/data/metadata.parquet"

# Preprocessing
SPARK_APP_NAME = "SemanticSearchStackExchange"
SPARK_MASTER = "local[*]"          # use all CPU cores on M2
SPARK_DRIVER_MEMORY = "8g"         # adjust based on your RAM (16GB recommended)

# Embedding model
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim, good for CPU

# FAISS index parameters
FAISS_INDEX_TYPE = "Flat"           # "Flat" for exact, "IVF100,PQ16" for approximate
FAISS_METRIC = "inner_product"      # cosine similarity via normalized vectors

# Search
DEFAULT_TOP_K = 10