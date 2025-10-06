import logging
import os
import sys
import csv
from pymilvus import Collection, MilvusClient, connections
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import APIClient, Credentials
from sentence_transformers import SentenceTransformer
from datetime import datetime
import time
from urllib.parse import urlparse
from pdfminer.high_level import extract_text as extract_pdf_text
import pdfplumber
import re
from pathlib import Path
from typing import List, Dict
import json
import tiktoken
from GLOBAL_keys import *

############### CONFIGURATION #################
collectionName = "UniMilano"
minChunkSize = 200
maxTokens = 512  # Limite del modello embedding
################################################

useWx = True
useLocal = True

modelIdParam = 'intfloat/multilingual-e5-large'
folderPath = 'scraped_texts/UNIMILANO/2025_10_01_18_00_00'

modelDimension = 1024 if modelIdParam == 'intfloat/multilingual-e5-large' else 384
model = None

# Inizializza tokenizer
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    tokenizer = None
    print("Warning: tiktoken not available, using character-based approximation")

if modelIdParam == 'intfloat/multilingual-e5-large':
    credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY_TZ)
    ai_client = APIClient(credentials)
    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: maxTokens,
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

# ============== UTILITY FUNCTIONS ==============

def clean_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def count_tokens(text: str) -> int:
    """Conta il numero di token in un testo"""
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Fallback: approssimazione (1 token ≈ 4 caratteri)
        return len(text) // 4

def truncate_text(text: str, max_tokens: int = 512) -> str:
    """Tronca il testo al numero massimo di token"""
    if not tokenizer:
        # Fallback: tronca per caratteri
        max_chars = max_tokens * 4
        return text[:max_chars]
    
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Tronca e decodifica
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)

def split_large_chunk(chunk_text: str, max_tokens: int = 512) -> List[str]:
    """
    Divide un chunk troppo grande in sotto-chunks
    rispettando il limite di token
    """
    token_count = count_tokens(chunk_text)
    
    if token_count <= max_tokens:
        return [chunk_text]
    
    # Dividi per paragrafi
    paragraphs = chunk_text.split('\n\n')
    sub_chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        test_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        if count_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                sub_chunks.append(current_chunk.strip())
            
            # Se anche un singolo paragrafo è troppo grande, troncalo
            if count_tokens(para) > max_tokens:
                sub_chunks.append(truncate_text(para, max_tokens))
                current_chunk = ""
            else:
                current_chunk = para
    
    if current_chunk:
        sub_chunks.append(current_chunk.strip())
    
    return sub_chunks

def EmbeddingText_WX(texts):
    try:
        embedding_vectors = embedding.embed_documents(texts=texts)
        return embedding_vectors[0]
    except Exception as e:
        logging.error(f"Watsonx embedding error: {e}")
        return None

def EmbeddingText_ST(texts):
    return model.encode(texts)

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

# ============== PDF TO MARKDOWN ==============

def pdf_to_markdown(pdf_path: str) -> str:
    """Converte un PDF in Markdown preservando la struttura"""
    markdown_lines = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.isupper() and len(line) > 15 and not line.isdigit():
                    markdown_lines.append(f"\n## {line}\n")
                elif re.match(r'^(Articolo|Art\.|Sezione|Capitolo)\s+\d+', line, re.IGNORECASE):
                    markdown_lines.append(f"\n### {line}\n")
                elif re.match(r'^\d+\.\d+', line):
                    markdown_lines.append(f"\n#### {line}\n")
                elif line.startswith(('•', '●', '◦', '-', '*')):
                    markdown_lines.append(f"{line}\n")
                elif re.match(r'^\d+\.\s+', line):
                    markdown_lines.append(f"{line}\n")
                elif re.match(r'^[a-z]\)\s+', line):
                    markdown_lines.append(f"{line}\n")
                else:
                    markdown_lines.append(f"{line}\n")
        
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    markdown_lines.append(convert_table_to_markdown(table))
                    markdown_lines.append("\n")
    
    markdown_content = ''.join(markdown_lines)
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
    return markdown_content

def convert_table_to_markdown(table):
    """Converte tabella in Markdown"""
    if not table or len(table) == 0:
        return ""
    
    markdown_table = []
    header = table[0]
    markdown_table.append("| " + " | ".join([str(cell or "") for cell in header]) + " |")
    markdown_table.append("|" + "|".join(["---" for _ in header]) + "|")
    
    for row in table[1:]:
        markdown_table.append("| " + " | ".join([str(cell or "") for cell in row]) + " |")
    
    return "\n".join(markdown_table)

# ============== CHUNKING STRATEGIES ==============

def chunk_by_articles(markdown_text: str, max_tokens: int = 1000) -> List[Dict]:
    """Divide il markdown per articoli con controllo token"""
    chunks = []
    
    articles = re.split(r'\n###\s+Articolo\s+\d+', markdown_text)
    article_titles = re.findall(r'###\s+(Articolo\s+\d+[^\n]*)', markdown_text)
    
    if articles[0].strip():
        header_text = articles[0].strip()
        
        # Controlla token dell'header
        if count_tokens(header_text) > max_tokens:
            sub_texts = split_large_chunk(header_text, max_tokens)
            for i, sub_text in enumerate(sub_texts):
                chunks.append({
                    'text': sub_text,
                    'metadata': {
                        'type': 'header_part',
                        'part_number': i + 1,
                        'article_number': 0,
                        'char_count': len(sub_text),
                        'token_count': count_tokens(sub_text)
                    }
                })
        else:
            chunks.append({
                'text': header_text,
                'metadata': {
                    'type': 'header',
                    'char_count': len(header_text),
                    'article_number': 0,
                    'token_count': count_tokens(header_text)
                }
            })
    
    for i, (title, content) in enumerate(zip(article_titles, articles[1:]), 1):
        full_text = f"### {title}\n{content.strip()}"
        
        # Controlla token dell'articolo
        if count_tokens(full_text) > max_tokens:
            sub_texts = split_large_chunk(full_text, max_tokens)
            for j, sub_text in enumerate(sub_texts):
                chunks.append({
                    'text': sub_text,
                    'metadata': {
                        'type': 'article_part',
                        'article_number': i,
                        'part_number': j + 1,
                        'article_title': title,
                        'char_count': len(sub_text),
                        'token_count': count_tokens(sub_text)
                    }
                })
        else:
            chunks.append({
                'text': full_text,
                'metadata': {
                    'type': 'article',
                    'article_number': i,
                    'article_title': title,
                    'char_count': len(full_text),
                    'token_count': count_tokens(full_text)
                }
            })
    
    return chunks

def chunk_generic_text(text: str, max_chunk_size: int = 1000, max_tokens: int = 512) -> List[Dict]:
    """Chunking generico con controllo token"""
    chunks = []
    
    custom_separator = re.search(r'\n([*]{5,}|[=]{5,}|[-]{5,})\n', text)
    
    if custom_separator:
        separator_pattern = custom_separator.group(1)
        sections = re.split(rf'\n{re.escape(separator_pattern)}\n', text)
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            lines = section.split('\n')
            title = lines[0].strip() if lines else f"Sezione {i+1}"
            
            # Verifica token
            if count_tokens(section) > max_tokens:
                # Dividi sezione in sotto-chunks
                sub_texts = split_large_chunk(section, max_tokens)
                for j, sub_text in enumerate(sub_texts):
                    chunks.append({
                        'text': sub_text,
                        'metadata': {
                            'type': 'section_part',
                            'section_number': i + 1,
                            'part_number': j + 1,
                            'title': title,
                            'char_count': len(sub_text),
                            'token_count': count_tokens(sub_text)
                        }
                    })
            else:
                chunks.append({
                    'text': section,
                    'metadata': {
                        'type': 'section',
                        'section_number': i + 1,
                        'title': title,
                        'char_count': len(section),
                        'token_count': count_tokens(section)
                    }
                })
    else:
        # Chunking per paragrafi con controllo token
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_number = 1
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if count_tokens(test_chunk) <= max_tokens and len(test_chunk) < max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'type': 'paragraph_group',
                            'chunk_number': chunk_number,
                            'char_count': len(current_chunk),
                            'token_count': count_tokens(current_chunk)
                        }
                    })
                    chunk_number += 1
                
                # Se paragrafo singolo supera max_tokens, troncalo
                if count_tokens(para) > max_tokens:
                    para = truncate_text(para, max_tokens)
                
                current_chunk = para
        
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'type': 'paragraph_group',
                    'chunk_number': chunk_number,
                    'char_count': len(current_chunk),
                    'token_count': count_tokens(current_chunk)
                }
            })
    
    return chunks

def generate_chunks(text: str, file_type: str = 'auto', max_tokens: int = 512) -> List[Dict]:
    """Genera chunks con strategia adattiva e controllo token"""
    if file_type == 'auto':
        if re.search(r'###\s+Articolo\s+\d+', text):
            file_type = 'articles'
        elif re.search(r'\n[*=]{5,}\n', text):
            file_type = 'sections'
        else:
            file_type = 'generic'
    
    if file_type == 'articles':
        return chunk_by_articles(text, max_tokens)
    else:
        return chunk_generic_text(text, max_chunk_size=800, max_tokens=max_tokens)

# ============== PROCESSING FUNCTIONS ==============

def ProcessTxtFile(documentPath, url, depth, language, documentType):
    """Processa file TXT con chunking intelligente e controllo token"""
    documentName = os.path.basename(documentPath)
    
    try:
        with open(documentPath, 'r', encoding='utf-8', errors='ignore') as f:
            content = clean_text(f.read())
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        return

    DeleteDocumentsInCollection(documentName)
    
    # Usa chunking intelligente con limite token
    try:
        chunk_objects = generate_chunks(content, file_type='auto', max_tokens=maxTokens)
        numChunks = len(chunk_objects)
        print(f"  Generated {numChunks} chunks")
    except Exception as e:
        print(f"  ✗ Error generating chunks: {e}")
        return
    
    inserted = 0
    skipped = 0
    
    for indexChunk, chunk_obj in enumerate(chunk_objects, start=1):
        chunk_text = chunk_obj['text']
        token_count = chunk_obj['metadata'].get('token_count', count_tokens(chunk_text))
        char_count = len(chunk_text)
        
        # Verifica dimensione minima
        if char_count < minChunkSize:
            print(f"  ⊘ Skipping chunk {indexChunk}/{numChunks} (too small: {char_count} chars)")
            skipped += 1
            continue
        
        # Verifica e correggi token
        if token_count > maxTokens:
            print(f"  ⚠ Chunk {indexChunk}/{numChunks} has {token_count} tokens (limit: {maxTokens}), truncating...")
            chunk_text = truncate_text(chunk_text, maxTokens)
            token_count = count_tokens(chunk_text)
            
            # Verifica di nuovo dopo il truncate
            if len(chunk_text) < minChunkSize:
                print(f"  ⊘ Skipping chunk {indexChunk}/{numChunks} (too small after truncation)")
                skipped += 1
                continue
        
        try:
            # Embedding
            embedding_result = (
                EmbeddingText_WX([chunk_text]) if useWx else EmbeddingText_ST([chunk_text])[0]
            )
            
            if embedding_result is None:
                print(f"  ✗ Failed to generate embedding for chunk {indexChunk}/{numChunks}")
                skipped += 1
                continue
            
            # Estrai metadata intelligenti
            section_info = chunk_obj['metadata'].get('title', chunk_obj['metadata'].get('type', 'content'))
            
            # Prepara dati per Milvus
            data = [{
                "docPath": folderPath,
                "documentName": documentName,
                "section": str(section_info)[:500],  # Limita anche section
                "link": url,
                "language": language,
                "depth": int(depth),
                "documentType": documentType,
                "contentChunk": chunk_text,
                "contentPage": truncate_text(content, 2000),
                "numChunks": numChunks,
                "indexChunk": indexChunk,
                "vectorDoc": embedding_result
            }]
            
            # Insert in Milvus
            milvusClient.insert(collection_name=collectionName, data=data)
            inserted += 1
            
            chunk_type = chunk_obj['metadata']['type']
            print(f"  ✓ Chunk {indexChunk}/{numChunks} inserted ({token_count} tokens, {char_count} chars, type: {chunk_type})")
        
        except Exception as e:
            print(f"  ✗ Error inserting chunk {indexChunk}/{numChunks}: {e}")
            skipped += 1
            continue
    
    print(f"  Summary: {inserted} inserted, {skipped} skipped out of {numChunks} chunks")
    

def ProcessPdfFile(documentPath, url, depth, language, documentType):
    return

    """Processa file PDF: conversione in MD + chunking intelligente"""
    documentName = os.path.basename(documentPath)
    
    try:
        # Converti PDF in Markdown
        print(f"  Converting PDF to Markdown...")
        markdown_text = pdf_to_markdown(documentPath)
        
        # Salva markdown (opzionale)
        #md_path = documentPath.replace('.pdf', '.md')
        #with open(md_path, 'w', encoding='utf-8') as f:
        #    f.write(markdown_text)
        #print(f"  Markdown saved: {os.path.basename(md_path)}")
        
    except Exception as e:
        print(f"  ✗ Error converting PDF to Markdown: {e}")
        return
    
    DeleteDocumentsInCollection(documentName)
    
    # Usa chunking intelligente con limite token
    try:
        chunk_objects = generate_chunks(markdown_text, file_type='auto', max_tokens=maxTokens)
        numChunks = len(chunk_objects)
        print(f"  Generated {numChunks} chunks")
    except Exception as e:
        print(f"  ✗ Error generating chunks: {e}")
        return
    
    inserted = 0
    skipped = 0
    
    for indexChunk, chunk_obj in enumerate(chunk_objects, start=1):
        chunk_text = chunk_obj['text']
        token_count = chunk_obj['metadata'].get('token_count', count_tokens(chunk_text))
        char_count = len(chunk_text)
        
        # Verifica dimensione minima
        if char_count < minChunkSize:
            print(f"  ⊘ Skipping chunk {indexChunk}/{numChunks} (too small: {char_count} chars)")
            skipped += 1
            continue
        
        # Verifica e correggi token
        if token_count > maxTokens:
            print(f"  ⚠ Chunk {indexChunk}/{numChunks} has {token_count} tokens (limit: {maxTokens}), truncating...")
            chunk_text = truncate_text(chunk_text, maxTokens)
            token_count = count_tokens(chunk_text)
            
            # Verifica di nuovo dopo il truncate
            if len(chunk_text) < minChunkSize:
                print(f"  ⊘ Skipping chunk {indexChunk}/{numChunks} (too small after truncation)")
                skipped += 1
                continue
        
        try:
            # Embedding
            embedding_result = (
                EmbeddingText_WX([chunk_text]) if useWx else EmbeddingText_ST([chunk_text])[0]
            )
            
            if embedding_result is None:
                print(f"  ✗ Failed to generate embedding for chunk {indexChunk}/{numChunks}")
                skipped += 1
                continue
            
            # Estrai metadata intelligenti
            section_info = chunk_obj['metadata'].get('article_title', 
                           chunk_obj['metadata'].get('title',
                           chunk_obj['metadata'].get('type', 'content')))
            
            # Prepara dati per Milvus
            data = [{
                "docPath": folderPath,
                "documentName": documentName,
                "section": str(section_info)[:500],  # Limita anche section
                "link": url,
                "language": language,
                "depth": int(depth),
                "documentType": documentType,
                "contentChunk": chunk_text,
                "contentPage": truncate_text(markdown_text, 2000),
                "numChunks": numChunks,
                "indexChunk": indexChunk,
                "vectorDoc": embedding_result
            }]
            
            # Insert in Milvus
            milvusClient.insert(collection_name=collectionName, data=data)
            inserted += 1
            
            chunk_type = chunk_obj['metadata']['type']
            print(f"  ✓ Chunk {indexChunk}/{numChunks} inserted ({token_count} tokens, {char_count} chars, type: {chunk_type})")
        
        except Exception as e:
            print(f"  ✗ Error inserting chunk {indexChunk}/{numChunks}: {e}")
            skipped += 1
            continue
    
    print(f"  Summary: {inserted} inserted, {skipped} skipped out of {numChunks} chunks")


# ============== MAIN ==============

if __name__ == "__main__":
    startProcess = datetime.now()
    numProcessed = 0
    numSkipped = 0

    csvPath = os.path.join(folderPath, "scraped_pages.csv")
    if not os.path.exists(csvPath):
        print(f"CSV file not found: {csvPath}")
        sys.exit(1)

    print(f"Starting processing at {startProcess}")
    print(f"Configuration: maxTokens={maxTokens}, minChunkSize={minChunkSize}")
    print(f"Using {'Watsonx' if useWx else 'SentenceTransformer'} for embeddings")
    print("=" * 60)

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

                print(f"\nProcessing: {fileName}")
                print(f"  URL: {url}")
                print(f"  Language: {language}")
                
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
                logging.error(f"Error processing {fileName}: {e}")
                import traceback
                traceback.print_exc()
                numSkipped += 1

    stopProcess = datetime.now()
    duration_minutes = (stopProcess - startProcess).seconds / 60
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Started:    {startProcess.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished:   {stopProcess.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:   {duration_minutes:.2f} minutes")
    print(f"Processed:  {numProcessed} files")
    print(f"Skipped:    {numSkipped} files")
    print(f"Success rate: {(numProcessed/(numProcessed+numSkipped)*100) if (numProcessed+numSkipped) > 0 else 0:.1f}%")
    print("=" * 60)