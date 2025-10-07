# Verwende ein stabileres und aktuelleres Debian-Basis-Image (Bullseye)
# Statt des veralteten 'buster', um 404-Fehler bei apt-get zu vermeiden.
FROM python:3.9-slim-bullseye

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Installiere PostgreSQL client development files needed for psycopg2
# Aktualisiert die Repositories und installiert libpq-dev
RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

# Kopiere die Abhängigkeitsdatei in das Arbeitsverzeichnis
COPY requirements.txt .

# Installiere Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den gesamten Anwendungscode in den Container
COPY . .

# Setze Umgebungsvariablen für die Anwendung (nicht kritisch, aber gute Praxis)
ENV PYTHONUNBUFFERED 1

# Das Entrypoint-Skript wird von docker-compose überschrieben, 
# aber wir definieren einen Standard.
CMD ["python", "protokoll_agent.py"]