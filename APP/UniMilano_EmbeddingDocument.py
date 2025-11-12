import logging
import os
import sys
import csv
from pymilvus import Collection, MilvusClient, connections
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import APIClient, Credentials
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import time
from urllib.parse import urlparse
import pdfplumber


# *************** INITIAL PARAMETERS ***************
#*** MILVUS CONFIG ***
MILVUS_URL = "http://localhost"
MILVUS_PORT = "19530"
MILVUS_HOST = "localhost"
MILVUS_ALIAS = "default"
COLLECTION_NAME = "samplecollection"
COLLECTION_DESCR = "Description for samplecollection"
DELETE_COLLECTION_IF_EXISTS = True

#*** WATSONX CONFIG ***
WX_URL = "https://us-south.ml.cloud.ibm.com"
WX_API_KEY = "fpMu8viNLafvrayEkNyXxKjyYWOz9rFqzFBbg47-6U2i"
WX_PROJECT_ID = '05a78408-7b0c-4c6f-ad34-5e267488200c'
WX_EMBEDING_MODEL = "intfloat/multilingual-e5-large"
WX_EMBEDING_MODEL_DIM = 1024

#*** DOCUMENT TO IMPORT CONFIG ***
DOCUMENT_FOLDER_PATH = "/Users/antoniomarinelli/Desktop/Python Mongodb/scraped_texts/UNIMILANO/2025_10_01_18_00_00"
DOCUMENT_CSV_FILE = "scraped_pages.csv"


collectionName = COLLECTION_NAME
minChunkSize = 200

useWx = True
useLocal = True

modelIdParam = WX_EMBEDING_MODEL

modelDimension = WX_EMBEDING_MODEL_DIM
model = None

credentials = Credentials(url=WX_URL, api_key=WX_API_KEY)
ai_client = APIClient(credentials)
embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
    EmbedParams.RETURN_OPTIONS: {'input_text': True}
}
embedding = Embeddings(
    model_id=modelIdParam,
    credentials=ai_client.credentials,
    params=embed_params,
    project_id=WX_PROJECT_ID,
    verify=False
)

connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
milvusClient = MilvusClient(uri=MILVUS_URL)

def clean_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def EmbeddingText_WX(texts):
    try:
        embedding_vectors = embedding.embed_documents(texts=texts)
        return embedding_vectors[0]
    except Exception as e:
        logging.error(f"Watsonx embedding error: {e}")
        return None

def EmbeddingText_ST(texts):
    return model.encode(texts)

def GetChunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return splitter.split_text(text)

def DeleteDocumentsInCollection(documentName):
    collection = Collection(collectionName)
    collection.delete(expr=f"documentName == '{documentName}'")
    collection.flush()
    time.sleep(2)

def extract_language_from_url(url):
    try:
        path_parts = urlparse(url).path.strip("/").split("/")
        if path_parts:
            lang = path_parts[0].lower()
            if lang in ["it", "en", "fr", "de", "es", "pt"]: 
                return lang
        return "unknown"
    except Exception:
        return "unknown"

def table_to_markdown(table):
    if not table or not table[0]:
        return ""
    
    table = [[cell if cell else "" for cell in row] for row in table]
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]
    
    markdown_rows = []
    header = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(table[0])) + " |"
    markdown_rows.append(header)
    
    separator = "|" + "|".join("-" * (width + 2) for width in col_widths) + "|"
    markdown_rows.append(separator)
    
    for row in table[1:]:
        row_str = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        markdown_rows.append(row_str)
    
    return "\n".join(markdown_rows)

def pdf_to_markdown(pdf_path):
    markdown_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables()
            
            if text:
                markdown_content.append(text)
            
            if tables:
                for table in tables:
                    markdown_content.append("\n" + table_to_markdown(table) + "\n")
    
    return "\n".join(markdown_content)

def ProcessTxtFile(documentPath, url, depth, language, documentType):
    documentName = os.path.basename(documentPath)
    with open(documentPath, 'r', encoding='utf-8', errors='ignore') as f:
        content = clean_text(f.read())

    DeleteDocumentsInCollection(documentName)
    chunks = [clean_text(c) for c in GetChunks(content)]
    numChunks = len(chunks)

    MAX_CONTENT_LENGTH = 64000
    content_page = content[:MAX_CONTENT_LENGTH] if len(content) > MAX_CONTENT_LENGTH else content
  
    for indexChunk, chunk in enumerate(chunks, start=1):
        if len(chunk) < minChunkSize:
            continue
        embedding_result = (
            EmbeddingText_WX([chunk]) if useWx else EmbeddingText_ST([chunk])[0]
        )
        data = [{
            "docPath": DOCUMENT_FOLDER_PATH,
            "documentName": documentName,
            "section": 'webpage',
            "link": url,
            "language": language,
            "depth": int(depth),
            "documentType": documentType,
            "contentChunk": chunk,
            "contentPage": content_page,
            "numChunks": numChunks,
            "indexChunk": indexChunk,
            "vectorDoc": embedding_result
        }]
        milvusClient.insert(collection_name=collectionName, data=data)

def ProcessPdfFile(documentPath, url, depth, language, documentType):
    documentName = os.path.basename(documentPath)
    markdown_text = pdf_to_markdown(documentPath)
    
    MAX_CONTENT_LENGTH = 64000
    content_page = markdown_text[:MAX_CONTENT_LENGTH] if len(markdown_text) > MAX_CONTENT_LENGTH else markdown_text
    
    
    DeleteDocumentsInCollection(documentName)
    chunks = [clean_text(c) for c in GetChunks(markdown_text)]
    numChunks = len(chunks)

    for indexChunk, chunk in enumerate(chunks, start=1):
        if len(chunk) < minChunkSize:
            continue

        embedding_result = (
            EmbeddingText_WX([chunk]) if useWx else EmbeddingText_ST([chunk])[0]
        )
        
        data = [{
            "docPath": DOCUMENT_FOLDER_PATH,
            "documentName": documentName,
            "section": 'pdf',
            "link": url,
            "language": language,
            "depth": int(depth),
            "documentType": documentType,
            "contentChunk": chunk,
            "contentPage": content_page,
            "numChunks": numChunks,
            "indexChunk": indexChunk,
            "vectorDoc": embedding_result
        }]
        milvusClient.insert(collection_name=collectionName, data=data)

def StartImportProcess():
    startProcess = datetime.now()
    numProcessed = 0
    numSkipped = 0
    
    print("Process started at " + str(startProcess))

    csvPath = os.path.join(DOCUMENT_FOLDER_PATH, DOCUMENT_CSV_FILE)
    if not os.path.exists(csvPath):
        print(f"CSV file not found: {csvPath}")
        sys.exit(1)

    with open(csvPath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                fileName = row["Text File"]
                url = row["URL"]
                depth = row.get("Depth", "0")
                language = extract_language_from_url(url)
                documentType = row["Content-Type"]

                if not fileName:
                    continue

                filePath = os.path.join(DOCUMENT_FOLDER_PATH, fileName)
                if not os.path.exists(filePath):
                    logging.warning(f"File not found: {filePath}")
                    numSkipped += 1
                    continue

                print(f"Processing {fileName} (URL: {url})...")
                ext = os.path.splitext(fileName)[1].lower()
                if ext == ".pdf":
                    ProcessPdfFile(filePath, url, depth, language, documentType)
                elif ext == ".txt":
                    ProcessTxtFile(filePath, url, depth, language, documentType)
                else:
                    logging.warning(f"Unsupported file type: {fileName}")
                    numSkipped += 1
                    continue

                numProcessed += 1

            except Exception as e:
                logging.error(f"Error processing row: {e}")
                numSkipped += 1

    stopProcess = datetime.now()
    print("*" * 40)
    print(f"Process started at: {startProcess}")
    print(f"Process ended at: {stopProcess}")
    print(f"Processed: {numProcessed} files")
    print(f"Skipped: {numSkipped} files")
    print(f"Duration: {(stopProcess - startProcess).seconds / 60:.2f} minutes")
    

StartImportProcess()
