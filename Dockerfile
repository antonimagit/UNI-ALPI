#FROM python:3.11-slim
#COPY APP/ /APP/
#COPY WEB/ /WEB/
#RUN pip3.11 install --no-cache-dir -r /APP/requirements.txt
#EXPOSE 8080
#WORKDIR /APP
#CMD ["python3.11", "UniMilano_Server.py", "true"]


FROM python:3.11-slim

ENV WATSONX_URL="https://us-south.ml.cloud.ibm.com"
ENV WATSONX_API_KEY_TZ="fpMu8viNLafvrayEkNyXxKjyYWOz9rFqzFBbg47-6U2i"
ENV WATSONX_PROJECTID_TZ="05a78408-7b0c-4c6f-ad34-5e267488200c"
ENV LLM_GEN_URL="https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
ENV REDIS_HOST="redis-app-shared"
ENV REDIS_PORT="6379"
ENV MILVUS_HOST="milvus-standalone"
ENV MILVUS_PORT="19530"
ENV MILVUS_PROTOCOL="http"
ENV DB_HOST="postgres"
ENV DB_NAME="CSMDemoConfig"
ENV DB_PORT="5432"
ENV DB_USER="postgres"
ENV DB_PASSWORD="postgres"

# Copia SOLO il requirements.txt prima
COPY APP/requirements.txt /APP/requirements.txt

# Installa dipendenze Python (questo layer viene cachato!)
RUN pip3.11 install --no-cache-dir -r /APP/requirements.txt

# Copia le cartelle nella root del container DOPO l'installazione
COPY APP/ /APP/
COPY WEB/ /WEB/

# Esponi la porta
EXPOSE 8080

# Imposta la directory di lavoro dove sta il codice
WORKDIR /APP

# Esegui lo script
CMD ["python3.11", "UniMilano_Server.py", "true"]