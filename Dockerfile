#FROM python:3.11-slim
#COPY APP/ /APP/
#COPY WEB/ /WEB/
#RUN pip3.11 install --no-cache-dir -r /APP/requirements.txt
#EXPOSE 8080
#WORKDIR /APP
#CMD ["python3.11", "UniMilano_Server.py", "true"]


FROM python:3.11-slim

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