# Verwende ein stabileres und aktuelleres Debian-Basis-Image (Bullseye)
# Statt des veralteten 'buster', um 404-Fehler bei apt-get zu vermeiden.
FROM python:3.9-slim-bullseye

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Pfad für die isolierte Installation der Python-Abhängigkeiten, um Konflikte
# und Überschreibungen durch Volume Mounts zu vermeiden.
ENV PYTHON_DEPENDENCIES_PATH=/usr/local/lib/python-deps

# Füge das bin-Verzeichnis des Abhängigkeitspfades zum System-PATH hinzu
ENV PATH=$PYTHON_DEPENDENCIES_PATH/bin:$PATH

# Setze den PYTHONPATH, damit Python die Module im isolierten Verzeichnis findet.
# Obwohl 'docker-compose run' dies manchmal ignoriert, ist es eine gute Praxis.
ENV PYTHONPATH=$PYTHON_DEPENDENCIES_PATH

# Erstelle das Verzeichnis für die Abhängigkeiten
RUN mkdir -p $PYTHON_DEPENDENCIES_PATH

# Installiere PostgreSQL client development files needed for psycopg2 and Tkinter
# Aktualisiert die Repositories und installiert libpq-dev und python3-tk
RUN apt-get update && apt-get install -y libpq-dev python3-tk && rm -rf /var/lib/apt/lists/*

# Kopiere die Abhängigkeitsdatei in das Arbeitsverzeichnis
COPY requirements.txt .

# Installiere Python-Abhängigkeiten in den isolierten Ordner
# Die --target Option stellt sicher, dass die Pakete an einem Ort landen,
# der nicht vom Volume Mount in docker-compose.yml betroffen ist.
RUN pip install --no-cache-dir --target=$PYTHON_DEPENDENCIES_PATH -r requirements.txt

# Kopiere den gesamten Anwendungscode in den Container
COPY . .

# Setze Umgebungsvariablen für die Anwendung (nicht kritisch, aber gute Praxis)
ENV PYTHONUNBUFFERED 1

# Ersetze die CMD-Anweisung, um PYTHONPATH explizit zu setzen.
# Dies ist der robuste Workaround, der sicherstellt, dass die Module gefunden werden,
# falls der Container ohne Überschreiben des Befehls ausgeführt wird.
CMD ["/bin/bash", "-c", "PYTHONPATH=$PYTHON_DEPENDENCIES_PATH python indexer.py"]