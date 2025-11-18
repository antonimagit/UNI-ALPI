from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import APIClient, Credentials
from pymilvus import Collection, MilvusClient, connections
import logging
import sys
import threading
import time
from GLOBAL_keys import *

authenticator = IAMAuthenticator(WATSONX_API_KEY_TZ)
_token_lock = threading.Lock()

def get_cached_token():
    with _token_lock:
        return authenticator.token_manager.get_token()
    
#YOUR_ACCESS_TOKEN = authenticator.token_manager.get_token()

auth_url = "https://iam.cloud.ibm.com/identity/token"
auth_headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json"
}
auth_data = {
    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
    "apikey": WATSONX_API_KEY_TZ
}

modelIdParam = 'intfloat/multilingual-e5-large'
numresultParam = 5

#MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
#MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

#connections.connect(
#    alias="default",
#    host=MILVUS_HOST,
#    port=MILVUS_PORT
#)
#milvusClient = MilvusClient(
#    uri="http://localhost:19530"
#)

try:
    credentials = Credentials(
        url=WATSONX_URL,
        api_key=WATSONX_API_KEY_TZ
    )
    ai_client = APIClient(credentials)
    logging.info("Connected to IBM Watson AI")
except Exception as e:
    logging.error(f"Failed to connect to IBM Watson AI: {e}")
    sys.exit(1)


model_id = modelIdParam
project_id = WATSONX_PROJECTID_TZ
embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: TRUNCATE_INPUT_TOKENS,
    EmbedParams.RETURN_OPTIONS: {'input_text': True}
}

# Initialize embedding model
embedding = Embeddings(
    model_id=model_id,
    credentials=ai_client.credentials,
    params=embed_params,
    project_id=project_id,
    space_id=None,
    verify=False
)
model = modelIdParam