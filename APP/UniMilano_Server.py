import os
from UniMilano_Config import *
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, render_template, redirect, url_for
from datetime import datetime
import psycopg2
import requests
from rapidfuzz import fuzz
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import redis
import json
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from deep_translator import GoogleTranslator
from GLOBAL_Assistant_IntentClassifier import create_classifier

HEADER = '\033[95m'
OKCYAN = '\033[96m'
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RESET = "\033[0m"

APP_PREFIX = "unimilano"
LLM_MODEL = "mistralai/mistral-medium-2505"

############### MILVUS CONFIGURATION #################
collectionName = "UniMilano"
minChunkSize = 20
################################################


def tracelog(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] -> {message}")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_PROTOCOL = "http"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "CSMDemoConfig"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres")
}

# Setup Redis
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    tracelog("✅ Redis connected successfully")
except Exception as e:
    tracelog(f"Redis connection failed: {e}")
    redis_client = None

tracelog("Start import Config")
tracelog("Importing Config...")
tracelog("✅ Finish import Config")

app = Flask(__name__,
            static_folder='../WEB',
            static_url_path='',
            template_folder='../WEB')

# Crea l'istanza del classificatore
try:
    print("1")
    intent_classifier = create_classifier(tracelog)
    print("7")
    tracelog("✅ Intent classifier initialized successfully")
except Exception as e:
    tracelog(f"CRITICAL ERROR: Failed to initialize intent classifier: {e}")
    sys.exit(1)

# Database connection helper


def get_db_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port="5432",
        database="CSMDemoConfig",
        user="postgres",
        password="postgres"
    )


# ================================================================
# MILVUS
# ================================================================

"""
def delete_data_from_milvus(filter_expression, flush=True):
    try:
        collection = Collection(collectionName)
        collection.delete(expr=filter_expression)
        if flush:
            collection.flush()
            time.sleep(0.5)
        print(f"✅ Eliminati record da '{collectionName}' con filtro: {filter_expression}")
    except Exception as e:
        print(f"❌ Errore durante l'eliminazione: {e}")
"""

def delete_data_from_milvus(filter_expression, flush=True):
    try:
        uriConfig = MILVUS_PROTOCOL + "://" + MILVUS_HOST + ":" + MILVUS_PORT
        milvusClient = MilvusClient(uri=uriConfig)
        milvusClient.delete(
            collection_name=collectionName,
            filter=filter_expression
        )
        if flush:
            milvusClient.flush(collection_name=collectionName)
            time.sleep(0.5)
        print(f"✅ Eliminati record da '{collectionName}' con filtro: {filter_expression}")
    except Exception as e:
        print(f"❌ Errore durante l'eliminazione: {e}")

def insert_data_to_milvus(link, contentPage):
    try:
        print("Processing: " + link)

        chunks = GetChunksFromText(contentPage)
        numChunks = len(chunks)
        indexChunk = 0

        for chunk in chunks:
            indexChunk += 1
            chunkText = chunk  # ATTENTION ###: if use the encoder from_tiktoken_encoder use chunk.page_content to get the text
            if (len(chunkText) < minChunkSize):
                continue

            embedding_result = EmbeddingText_WX([chunkText])
            if not embedding_result or not embedding_result[0]:
                continue

            ### INSERT THE EMBEDDING IN MILVUS VECTOR DB  ###
            data = [
                {
                    "docPath": '',
                    "documentName": link.strip(),
                    "section": 'custom',
                    "link": link.strip(),
                    "language": 'it',
                    "depth": 0,
                    "documentType": 'html',
                    "contentChunk": chunk,
                    "contentPage": contentPage,
                    "numChunks": numChunks,
                    "indexChunk": indexChunk,
                    "vectorDoc": embedding_result
                }
            ]
            # INSERISCO I DATI IN MILVUS
            uriConfig = MILVUS_PROTOCOL + "://" + MILVUS_HOST + ":" + MILVUS_PORT
            milvusClient = MilvusClient(uri=uriConfig)
            res = milvusClient.insert(collection_name=collectionName, data=data)
    except Exception as e:
        print(f"❌ Errore durante l'inserimento: {e}")


def EmbeddingText_WX(texts):
    try:
        embedding_vectors = embedding.embed_documents(texts=texts)
        return embedding_vectors[0]
    except Exception as e:
        logging.error(f"Error embedding texts with watsonx: {e}")
        return None
    
def GetChunksFromText(pageText, context=""):
    pageText = pageText.replace("\n\n", "\n")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(pageText)
    prefix = context + "\n" if context else ""
    return [f"{prefix}{chunk}" for chunk in chunks]


# =================================================================
# ROUTE FRONTEND
# =================================================================

@app.route('/')
def home():
    return render_template('unimilano.html')


@app.route('/unimilanoconfig')
def unimilano_config():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            'SELECT * FROM public."UNIMILANO_BlackList" ORDER BY "idBlackList" ASC')
        blacklist = cur.fetchall()

        cur.execute(
            'SELECT * FROM public."UNIMILANO_Insights" ORDER BY "ID_insight" ASC')
        insights = cur.fetchall()

        cur.close()
        conn.close()

        # Converti tuple in dict per il template
        blacklist_data = [
            {
                'idBlackList': row[0],
                'Question': row[1],
                'Answer': row[2],
                'threshold': row[3]
            } for row in blacklist
        ]

        insights_data = [
            {
                'ID_insight': row[0],
                'Insights': row[1],
                'LinkRisorsa': row[2]
            } for row in insights
        ]

        return render_template('UnimilanoConfig.html',
                               blacklist=blacklist_data,
                               insights=insights_data)
    except Exception as e:
        tracelog(f"Error in /unimilanoconfig: {e}")
        return "Errore caricamento dati", 500


@app.route('/emailtemplateunimilano/update', methods=['POST'])
def update_email_template():
    id_email_template = request.form.get('idEmailTemplate')
    mail_template = request.form.get('MailTemplate')

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if id_email_template:
            cur.execute(
                'UPDATE public."UNIMILANO_EmailTemplate" SET "MailTemplate" = %s WHERE "idEmailTemplate" = %s',
                (mail_template, id_email_template)
            )
        else:
            cur.execute(
                'INSERT INTO public."UNIMILANO_EmailTemplate" ("MailTemplate") VALUES (%s)',
                (mail_template,)
            )

        conn.commit()
        cur.close()
        conn.close()

        return redirect('/unimilanoconfig')
    except Exception as e:
        tracelog(f"Error updating email template: {e}")
        return "Errore aggiornamento template", 500


@app.route('/blacklistunimilano/add', methods=['POST'])
def add_blacklist():
    question = request.form.get('question')
    answer = request.form.get('answer')
    threshold = request.form.get('threshold')

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            'INSERT INTO public."UNIMILANO_BlackList" ("Question", "Answer", "threshold") VALUES (%s, %s, %s)',
            (question, answer, threshold)
        )

        conn.commit()
        cur.close()
        conn.close()

        return redirect('/unimilanoconfig')
    except Exception as e:
        tracelog(f"Error adding blacklist: {e}")
        return "Errore inserimento", 500


@app.route('/blacklistunimilano/delete/<int:id>', methods=['POST'])
def delete_blacklist(id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            'DELETE FROM public."UNIMILANO_BlackList" WHERE "idBlackList" = %s', (id,))

        conn.commit()
        cur.close()
        conn.close()

        return redirect('/unimilanoconfig')
    except Exception as e:
        tracelog(f"Error deleting blacklist: {e}")
        return "Errore eliminazione", 500


@app.route('/insightsunimilano/add', methods=['POST'])
def add_insight():
    insight = request.form.get('insight')
    link_risorsa = request.form.get('LinkRisorsa')

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            'INSERT INTO public."UNIMILANO_Insights" ("Insights", "LinkRisorsa") VALUES (%s, %s)',
            (insight, link_risorsa)
        )

        conn.commit()
        cur.close()
        conn.close()

        # Chiama il custom embedding (devi avere questa funzione)
        insert_data_to_milvus(link_risorsa, insight)

        return redirect('/unimilanoconfig')
    except Exception as e:
        tracelog(f"Error adding insight: {e}")
        return "Errore inserimento insight", 500


@app.route('/insightsunimilano/delete/<int:id>', methods=['POST'])
def delete_insight(id):
    link_risorsa = request.form.get('LinkRisorsa')

    try:
        tracelog(f"idP: {id}")
        tracelog(f"LinkRisorsa: {link_risorsa}")
        
        deleteExpression = f"link == '{link_risorsa}'"
        delete_data_from_milvus(deleteExpression, True)

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            'DELETE FROM public."UNIMILANO_Insights" WHERE "ID_insight" = %s', (id,))

        conn.commit()
        cur.close()
        conn.close()

        return redirect('/unimilanoconfig')
    except Exception as e:
        tracelog(f"Error deleting insight: {e}")
        return "Errore eliminazione insight", 500


@app.route("/query", methods=["POST"])
def run_script():
    # ... tutto il tuo codice esistente per /query ...
    tracelog("Start query process")
    startProcess = datetime.now()
    data = request.json
    tracelog(RED + "=" * 70 + RESET)
    tracelog(f"{RED}START NEW CHAT{RESET}")
    tracelog(RED + "=" * 70 + RESET)

    query = str(data.get("input_text", ""))
    query = query.replace('[', '').replace(']', '')
    tracelog('Query:' + str(query))

    lang = detect_language(query)
    tracelog("Detected language: " + lang)
    isTraslated = False
    originalQuery = query
    italianResponse = None

    if (lang != 'it'):
        isTraslated = True
        query = translate_text(query, "it")
        tracelog("Query translated: " + query)

    custom_session_id = str(data.get("custom_session_id", ""))
    tracelog('custom_session_id:' + str(custom_session_id))

    records = GetForbiddenQuestions()

    if records != "error":
        for row in records:
            if is_similar_text(row[1], query, row[2]):
                tracelog("Domanda in Black List!")
                df = row[3]
                return jsonify({"resultGenAI": df, "error": "Errore durante check Black List."})

    conversation_history = get_conversation_history(custom_session_id)
    tracelog(
        f"Retrieved {len(conversation_history)} previous conversation turns")

    filterSearch = str(data.get("filterSearch", ""))
    filterSearch = filterSearch.replace('[', '').replace(']', '')
    tracelog('query filterSearch:' + str(filterSearch))

    documentChunks = ''
    documentLinks = ''

    intent = intent_classifier.classify_query_intent(
        query, conversation_history)
    pattern_info = intent_classifier.last_matched_pattern
    tracelog(f"✅ Query intent classified as: {intent}")

    useCacheInLLM = False
    saveInCache = True
    refined_query = ''

    if intent == 'FOLLOW_UP':
        refined_query = analyze_and_refine_query(query, conversation_history)
        documentChunks, documentLinks = ProcessDocSearch(
            refined_query, filterSearch)
        useCacheInLLM = True
    else:
        tracelog("New query: searching on Milvus")
        documentChunks, documentLinks = ProcessDocSearch(query, filterSearch)

    if (documentChunks == 'null'):
        saveInCache = False
        resultGenAI = "Mi dispiace ma non sono riuscito a trovare la risposta alla tua domanda."
    else:
        tracelog("Start generate response in LLM")
        query_to_use = refined_query if useCacheInLLM else query
        tracelog("Query per l'LLM: " + query_to_use)
        resultGenAI = GenerateResponse(
            query_to_use, documentChunks, conversation_history, useCacheInLLM)

        noresultstring = "non sono riuscito a trovare la risposta alla tua domanda"

        if noresultstring not in resultGenAI:
            resultGenAI += '<br><br>Risorse:<br>' + documentLinks
        tracelog("✅ Finish generate response in LLM")

    stopProcess = datetime.now()
    duration = stopProcess - startProcess

    if (resultGenAI == "Mi dispiace ma non sono riuscito a trovare la risposta alla tua domanda."):
        saveInCache = False

    if (isTraslated):
        italianResponse = resultGenAI
        if '<br><br>Risorse:<br>' in resultGenAI:
            parts = resultGenAI.split('<br><br>Risorse:<br>')
            text_part = translate_text(parts[0], lang)
            resultGenAI = text_part + '<br><br>Risorse:<br>' + parts[1]
        else:
            resultGenAI = translate_text(resultGenAI, lang)

    if custom_session_id and saveInCache == True:
        save_conversation_turn(custom_session_id, query, resultGenAI, intent,
                               pattern_info, isTraslated, originalQuery, italianResponse)

    tracelog('Duration: ' + str(duration))

    return jsonify({"resultGenAI": resultGenAI, "error": resultGenAI})


def GetForbiddenQuestions():
    try:
        psConnection = psycopg2.connect(**DB_CONFIG)
        psCursor = psConnection.cursor()

        query = """
            SELECT * FROM public."UNIMILANO_BlackList"
        """
        psCursor.execute(query)
        result = psCursor.fetchall()

        psCursor.close()
        psConnection.close()

        return result
    except Exception as e:
        tracelog("Errore:", e)
        return "error"


def is_similar_text(q1, q2, threshold=0.8):
    return SequenceMatcher(None, q1.lower(), q2.lower()).ratio() >= threshold


def EmbeddingText_WX(texts):
    try:
        embedding_vectors = embedding.embed_documents(texts=texts)
        return embedding_vectors[0]
    except Exception as e:
        logging.error(f"Error embedding texts with watsonx: {e}")
        return None


def GenerateResponse(input_text, documentChunks, conversation_history=[], useCacheInLLM=False):

    strPrompt = ''
    if conversation_history and useCacheInLLM:
        strPrompt += 'Hai tre input: cronologia conversazione, testo di riferimento e domanda.\n'
        strPrompt += 'Rispondi utilizzando principalmente le conversazioni precedenti e se necessario anche il testo di riferimento.\n'
        strPrompt += '\nCronologia conversazioni precedenti:\n'
        for turn in conversation_history:
            strPrompt += f"Utente: {turn['user_query']}\n"
            strPrompt += f"Assistente: {turn['assistant_response']}\n\n"
        strPrompt += '---\n\n'
    else:
        strPrompt += 'Hai due input: testo di riferimento e domanda.\n'
        strPrompt += 'Rispondi utilizzando solo le informazioni che trovi nel testo di riferimento.\n'

    strPrompt += 'Rispondi in maniera colloquiale e dettagliata.\n'
    strPrompt += 'Non inventare nulla e non includere la domanda nella risposta.\n'
    if conversation_history and useCacheInLLM:
        strPrompt += 'Se la risposta non è presente nelle conversazioni precedenti e nel testo di riferimento, scrivi: "Mi dispiace ma non sono riuscito a trovare la risposta alla tua domanda".\n'
    else:
        strPrompt += 'Se la risposta non è presente nel testo, scrivi: "Mi dispiace ma non sono riuscito a trovare la risposta alla tua domanda".\n'
    strPrompt += '*** INIZIO TESTO DI RIFERIMENTO: ***\n'
    strPrompt += documentChunks.strip() + '\n'
    strPrompt += '*** FINE TESTO DI RIFERIMENTO ***\n'
    strPrompt += 'Domanda:\n'
    strPrompt += input_text.strip() + '\n'
    strPrompt += 'Risposta:'

    tracelog('=' * 70)
    tracelog(strPrompt)
    tracelog('=' * 70)

    body = {
        "input": strPrompt,
        "parameters": {
            "decoding_method": 'greedy',  # (Use 'sample' or 'greedy')
            "max_new_tokens": 6000,
            "min_new_tokens": 0,
            "temperature": 0,
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": 1
        },
        "model_id": LLM_MODEL,
        "project_id": WATSONX_PROJECTID_TZ
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + get_cached_token()
    }

    # Make the request with stream=True
    response = requests.post(
        LLM_GEN_URL,
        headers=headers,
        json=body,
        stream=True
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    genResponse = data.get('results', [])[0].get('generated_text', '').strip()

    return (genResponse)


def detect_language(user_text):
    from langdetect import detect
    try:
        lang = detect(user_text)
        return lang
    except:
        tracelog("ERROR IN DETECTION LANGUAGE")
        return 'it'  # Default italiano se fallisce


def translate_text(text, target_lang='it'):
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        return translated
    except:
        return text  # Ritorna originale se fallisce


def analyze_and_refine_query(user_query, conversation_history):
    """
    Analizza la query utente e identifica a quale domanda precedente si riferisce,
    poi genera una query raffinata per Milvus
    """

    # Costruisci il prompt con le domande precedenti
    previous_questions = []
    for i, turn in enumerate(conversation_history):
        previous_questions.append(f"{i+1}. {turn['user_query']}")

    questions_list = '\n'.join(previous_questions)

    strPrompt = f"""L'utente sta facendo una domanda di follow-up. Analizza a quale domanda precedente si riferisce e genera una nuova query ottimizzata per la ricerca.

Domande precedenti (dalla più vecchia alla più recente):
{questions_list}

Domanda follow-up dell'utente: {user_query}

Ragiona così:
1. Esamina le parole chiave nel follow-up per capire l'argomento
2. Se non ci sono riferimenti specifici, assumi che si riferisca alla domanda più recente (l'ultima nella lista)
3. Combina la domanda identificata con la richiesta specifica del follow-up
4. Genera una nuova query che combini la domanda originale con la richiesta di follow-up
5. Rispondi SOLO con la nuova query, nient'altro

Nuova query:"""

    body = {
        "input": strPrompt,
        "parameters": {
            "decoding_method": 'greedy',
            "max_new_tokens": 200,
            "min_new_tokens": 0,
            "temperature": 0,
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": 1
        },
        "model_id": LLM_MODEL,
        "project_id": WATSONX_PROJECTID_TZ
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + get_cached_token()
    }

    # tracelog("****** PROMPT ******")
    # tracelog(strPrompt)
    response = requests.post(
        LLM_GEN_URL, headers=headers, json=body, stream=True)

    if response.status_code != 200:
        tracelog(f"Error in analyze_and_refine_query: {response.text}")
        return user_query  # Fallback alla query originale

    data = response.json()
    refined_query = data.get('results', [])[0].get(
        'generated_text', '').strip()

    tracelog(f"Original query: {user_query}")
    tracelog(f"Refined query: {refined_query}")

    return refined_query if refined_query else user_query


def ProcessDocSearch(userChat, filterSearch):
    try:
        modelDimension = 1024
        embeddings = EmbeddingText_WX([userChat])
        if not connections.has_connection("default"):
            connections.connect(
                alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection("UniMilano")
        collection.load()
        query_vectors = [embeddings]
        # search_params = {"metric_type": "IP", "params": {"nprobe": 150}}
        search_params = {"metric_type": "COSINE"}

        if filterSearch != "":
            filter_expression = f'link == "{filterSearch}"'
            tracelog("filter_expression: " + filter_expression)
            results = collection.search(
                query_vectors, "vectorDoc", search_params, limit=1,
                output_fields=[
                    "id", "docPath", "documentName", "section", "link", "language",
                    "depth", "documentType", "contentChunk", "contentPage",
                    "numChunks", "indexChunk"
                ],
                expr=filter_expression
            )
        else:

            results = collection.search(
                query_vectors, "vectorDoc", search_params, limit=3,
                output_fields=[
                    "id", "docPath", "documentName", "section", "link", "language",
                    "depth", "documentType", "contentChunk", "contentPage",
                    "numChunks", "indexChunk"
                ]
            )

        documentChunks = ''
        documentLinks = ''

        for item in results[0]:
            contentChunk = item.fields["contentPage"]

            if contentChunk not in documentChunks:
                documentChunks += contentChunk + '\n\n'

            linkVar = item.fields["link"] + '\n'
            if linkVar not in documentLinks:
                documentLinks += linkVar

        return documentChunks, documentLinks

    except Exception as e:
        return "null", f"Error during the query in Milvus: {str(e)}"


def get_conversation_history(session_id, max_turns=5):
    """Recupera la cronologia conversazionale da Redis"""
    if not redis_client:
        return []

    try:
        # Usa il prefisso dell'app
        history_key = f"{APP_PREFIX}:conversation:{session_id}"
        history_json = redis_client.get(history_key)

        if history_json:
            history = json.loads(history_json)
            return history[-max_turns:] if len(history) > max_turns else history
        return []
    except Exception as e:
        tracelog(f"Error retrieving conversation history: {e}")
        return []


def save_conversation_turn(session_id, user_query, assistant_response, intent=None, followup_pattern=None, queryTraslated=False, originalQuery=None, italianResponse=None):
    """Salva un nuovo scambio conversazionale in Redis con cache chunks"""
    if not redis_client:
        return

    try:
        res = assistant_response.split("<br><br>Risorse:")[0]
        history_key = f"{APP_PREFIX}:conversation:{session_id}"
        existing_history = get_conversation_history(session_id, max_turns=50)

        new_turn = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "assistant_response": res,
            "intent": intent,
            "followup_pattern": followup_pattern,
            "queryTraslated": queryTraslated,
            "originalQuery": originalQuery,
            "italianResponse": italianResponse
        }

        existing_history.append(new_turn)

        # Mantieni solo gli ultimi 10 turn con cache
        # Gestione intelligente memoria conversazione
        max_total_turns = 20  # Limite totale conversazione

        # Rimuovi completamente turn più vecchi
        if len(existing_history) > max_total_turns:
            existing_history = existing_history[-max_total_turns:]

        redis_client.setex(history_key, 3600, json.dumps(existing_history))
        tracelog(
            f"Conversation turn saved for session {session_id} (intent: {intent})")

    except Exception as e:
        tracelog(f"Error saving conversation turn: {e}")


if __name__ == "__main__":
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=8080, debug=False)