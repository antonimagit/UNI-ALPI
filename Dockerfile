FROM python:3.11-slim

# Copia le cartelle nella root del container
COPY APP/ /APP/
COPY WEB/ /WEB/

# Installa dipendenze Python
RUN pip3.11 install --no-cache-dir -r /APP/requirements.txt

# Esponi la porta
EXPOSE 8080

# Imposta la directory di lavoro dove sta il codice
WORKDIR /APP

# Esegui lo script
CMD ["python3.11", "UniMilano_Server.py", "true"]