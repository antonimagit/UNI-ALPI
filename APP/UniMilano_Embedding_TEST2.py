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
import re
from GLOBAL_keys import *

############### CONFIGURATION #################
collectionName = "UniMilano"
minChunkSize = 200
################################################

useWx = True
useLocal = True

modelIdParam = 'intfloat/multilingual-e5-large'
folderPath = 'scraped_texts/UNIMILANO/2025_10_01_18_00_00'

modelDimension = 1024 if modelIdParam == 'intfloat/multilingual-e5-large' else 384
model = None

if modelIdParam == 'intfloat/multilingual-e5-large':
    credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY_TZ)
    ai_client = APIClient(credentials)
    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 1024,
        EmbedParams.RETURN_OPTIONS: {'input_text': True}
    }
    embedding = Embeddings(
        model_id=modelIdParam,
        credentials=ai_client.credentials,
        params=embed_params,
        project_id=WATSONX_PROJECTID_TZ,
        verify=False
    )
else:
    model = SentenceTransformer(modelIdParam)

connections.connect(alias="default", host='localhost', port='19530')
milvusClient = MilvusClient(uri="http://localhost:19530")

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
    """Converte tabella in markdown"""
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
    """Converte PDF in markdown con tabelle"""
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

    for indexChunk, chunk in enumerate(chunks, start=1):
        if len(chunk) < minChunkSize:
            continue
        embedding_result = (
            EmbeddingText_WX([chunk]) if useWx else EmbeddingText_ST([chunk])[0]
        )
        data = [{
            "docPath": folderPath,
            "documentName": documentName,
            "section": 'webpage',
            "link": url,
            "language": language,
            "depth": int(depth),
            "documentType": documentType,
            "contentChunk": chunk,
            "contentPage": content,
            "numChunks": numChunks,
            "indexChunk": indexChunk,
            "vectorDoc": embedding_result
        }]
        milvusClient.insert(collection_name=collectionName, data=data)

def ProcessPdfFile(documentPath, url, depth, language, documentType):
    documentName = os.path.basename(documentPath)
    
    # Converti PDF in markdown
    markdown_text = pdf_to_markdown(documentPath)
    
    DeleteDocumentsInCollection(documentName)
    
    # Chunking del markdown
    chunks = [clean_text(c) for c in GetChunks(markdown_text)]
    numChunks = len(chunks)

    for indexChunk, chunk in enumerate(chunks, start=1):
        if len(chunk) < minChunkSize:
            continue
        
        # Embedding del markdown
        embedding_result = (
            EmbeddingText_WX([chunk]) if useWx else EmbeddingText_ST([chunk])[0]
        )
        
        data = [{
            "docPath": folderPath,
            "documentName": documentName,
            "section": 'pdf',
            "link": url,
            "language": language,
            "depth": int(depth),
            "documentType": documentType,
            "contentChunk": chunk,
            "contentPage": markdown_text,
            "numChunks": numChunks,
            "indexChunk": indexChunk,
            "vectorDoc": embedding_result
        }]
        milvusClient.insert(collection_name=collectionName, data=data)

if __name__ == "__main__":
    startProcess = datetime.now()
    numProcessed = 0
    numSkipped = 0

    csvPath = os.path.join(folderPath, "scraped_pages.csv")
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

                filePath = os.path.join(folderPath, fileName)
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