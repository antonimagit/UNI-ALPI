
import os

WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY_TZ = os.getenv("WATSONX_API_KEY_TZ", "fpMu8viNLafvrayEkNyXxKjyYWOz9rFqzFBbg47-6U2i")
WATSONX_PROJECTID_TZ = os.getenv("WATSONX_PROJECTID_TZ", "05a78408-7b0c-4c6f-ad34-5e267488200c")
LLM_GEN_URL = os.getenv("LLM_GEN_URL", "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29")

REDIS_HOST = os.getenv("REDIS_HOST", "redis-app-shared")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_PROTOCOL = os.getenv("MILVUS_PROTOCOL", "http")