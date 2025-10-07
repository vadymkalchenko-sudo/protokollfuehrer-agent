import os
import logging
import sys
import json
import time # Für die Wartefunktion
import hashlib
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2 import sql 
from urllib.parse import urlparse 

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Environment Variable Loading ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_CONNECTION_URI = os.getenv("DB_CONNECTION_URI")

# --- Globale Initialisierung und Verbindung ---
conn = None
# NEUER FIX: Verwendet den realistischeren Supabase Dateinamen, der oft heruntergeladen wird.
SUPABASE_CERT_PATH = os.path.join(os.path.dirname(__file__), 'prod-ca-2021.crt')

def calculate_file_hash(filepath: str) -> str:
    """Berechnet den SHA256-Hash des Dateiinhalts und gibt ihn als Hex-String zurück."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def initialize_clients():
    """Initialisiert Gemini API und Datenbankverbindung."""
    global conn
    
    # --- Validation ---
    if not all([GEMINI_API_KEY, DB_CONNECTION_URI]):
        logging.error("🚨 FEHLER: Die kritischen Umgebungsvariablen GEMINI_API_KEY oder DB_CONNECTION_URI fehlen.")
        logging.error("HINWEIS: Bitte stellen Sie sicher, dass die .env-Datei korrekt befüllt ist.")
        sys.exit(1)
        
    # Prüfen, ob das Zertifikat existiert
    if not os.path.exists(SUPABASE_CERT_PATH):
        logging.error(f"🚨 KRITISCHER FEHLER: SSL Root Zertifikat nicht gefunden unter: {SUPABASE_CERT_PATH}")
        logging.error("BITTE: Laden Sie die Supabase CA Root Datei herunter und speichern Sie sie als 'prod-ca-2021.crt' im Projekt-Wurzelverzeichnis.")
        sys.exit(1)
        
    try:
        # 1. Gemini API global konfigurieren
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("✅ Gemini API erfolgreich konfiguriert.")
        
        # 2. PostgreSQL Verbindung (Parsen der Transaction Pooler URI)
        parsed_uri = urlparse(DB_CONNECTION_URI)
        
        # Verbindung mit voller SSL-Prüfung
        conn = psycopg2.connect(
            host=parsed_uri.hostname,
            port=parsed_uri.port,
            database=parsed_uri.path.lstrip('/'),
            user=parsed_uri.username,
            password=parsed_uri.password,
            # FIX: Erzwingt volle SSL-Verifizierung (Prüft Zertifikat)
            sslmode='verify-full', 
            sslrootcert=SUPABASE_CERT_PATH 
        )
        register_vector(conn)
        logging.info("✅ Datenbankverbindung über Pooler URI mit SSL/Zertifikatprüfung hergestellt.")
        return conn
    except Exception as e:
        logging.error(f"🚨 KRITISCHER FEHLER bei der Initialisierung oder Datenbankverbindung: {e}")
        logging.error("HINWEIS: Möglicherweise liegt der Fehler noch an der 'Address not in tenant allow_list' (siehe Supabase Console und Ihre IP-Adresse dort eintragen).")
        sys.exit(1)


# MODELL: Auf den empfohlenen Standard "text-embedding-004" umgestellt.
def get_embedding(text: str, model: str = "text-embedding-004") -> list[float]:
    """
    Generiert ein Vektor-Embedding für den gegebenen Text mit eingebauter Retry-Logik.
    """
    snippet = text[:60].replace('\n', ' ')
    logging.info(f"-> Generiere Embedding für Textausschnitt: '{snippet}...'")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = genai.embed_content( 
                model=model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return response['embedding']
        
        except Exception as e:
            error_message = str(e)
            
            if "429" in error_message or "Quota exceeded" in error_message or "Resource has been exhausted" in error_message:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logging.warning(f"⚠️ Rate Limit erreicht (429/Exhausted). Warte {wait_time}s vor Wiederholung ({attempt + 1}/{max_retries}).")
                    time.sleep(wait_time)
                else:
                    logging.error(f"🚨 FEHLER beim Generieren des Embeddings: Limit auch nach {max_retries} Versuchen überschritten. Tageslimit wahrscheinlich erreicht.")
                    return [] 
            else:
                logging.error(f"🚨 UNBEKANNTER FEHLER beim Generieren des Embeddings: {e}")
                return []
            
    return [] 

def load_protocol_data(filepath: str) -> Optional[dict]:
    """Lädt Protokolldaten aus einer Datei und erstellt Metadaten."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        filename = os.path.basename(filepath)
        meeting_id = os.path.splitext(filename)[0]

        return {
            "text": text,
            "metadata": { "source_file": filepath, "meeting_id": meeting_id }
        }
    except Exception as e:
        logging.error(f"Fehler beim Laden der Protokolldatei {filepath}: {e}")
        return None

def get_stored_hash(conn, source_file: str) -> Optional[str]:
    """Holt den gespeicherten content_hash für eine bestimmte source_file."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT metadata->>'content_hash' FROM protokolle WHERE metadata->>'source_file' = %s", (source_file,))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        logging.error(f"🚨 FEHLER beim Abrufen des Hashes für '{source_file}': {e}")
        return None

def delete_protocol(conn, source_file: str) -> None:
    """Löscht einen Protokolleintrag basierend auf der source_file."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM protokolle WHERE metadata->>'source_file' = %s",
                (source_file,)
            )
            conn.commit()
            logging.info(f"🗑️ Alten Eintrag für '{source_file}' gelöscht.")
    except Exception as e:
        logging.error(f"🚨 FEHLER beim Löschen des alten Eintrags für '{source_file}': {e}")
        conn.rollback()

def store_protocol(conn, protocol_text: str, embedding: list[float], metadata: dict) -> None:
    """
    Speichert Protokolltext, Embedding und Metadaten in der 'protokolle' Tabelle.
    """
    if not embedding:
        logging.warning("Skipping storage due to empty embedding.")
        return

    try:
        metadata_json = json.dumps(metadata)
        # FIX: pgvector benötigt eckige Klammern [] anstelle von geschweiften Klammern {}
        embedding_str = '[' + ', '.join(map(str, embedding)) + ']'

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("INSERT INTO protokolle (text, embedding, metadata) VALUES (%s, %s, %s)"),
                (protocol_text, embedding_str, metadata_json)
            )
            conn.commit()
            logging.info(f"✅ Protokoll erfolgreich gespeichert. Meeting ID: {metadata.get('meeting_id')}")
            
    except Exception as e:
        logging.error(f"🚨 FEHLER beim Speichern in die Datenbank: {e}")
        logging.warning("HINWEIS: Haben Sie die 'protokolle'-Tabelle und die 'pgvector'-Extension in Supabase erstellt?")
        conn.rollback() 


def main():
    """
    Hauptfunktion zur Verarbeitung und Indexierung von Besprechungsprotokollen aus einem Verzeichnis.
    """
    logging.info("=" * 40)
    logging.info("Protokoll Indexing Agent - Start")
    logging.info("=" * 40)
    
    # Initialisiert Clients und bricht bei Fehler ab
    conn = initialize_clients()
    
    protocols_dir = "protocols"
    if not os.path.isdir(protocols_dir):
        logging.error(f"Fehler: Das Verzeichnis '{protocols_dir}' wurde nicht gefunden.")
        sys.exit(1)

    protocol_files = [os.path.join(protocols_dir, f) for f in os.listdir(protocols_dir) if os.path.isfile(os.path.join(protocols_dir, f))]

    for filepath in protocol_files:
        protocol_data = load_protocol_data(filepath)
        if not protocol_data:
            continue

        source_file = protocol_data["metadata"]["source_file"]
        protocol_text = protocol_data["text"]

        # 1. Aktuellen Hash berechnen
        current_hash = calculate_file_hash(filepath)
        protocol_data["metadata"]["content_hash"] = current_hash

        # 2. Gespeicherten Hash aus der DB holen
        stored_hash = get_stored_hash(conn, source_file)

        # 3. Logik zur (Neu-)Indizierung
        if stored_hash == current_hash:
            logging.info(f"Skipping '{source_file}': Content is unchanged (Hash match).")
            continue

        if stored_hash is not None:
            logging.info(f"Content of '{source_file}' has changed. Deleting old entry and re-indexing...")
            delete_protocol(conn, source_file)
        else:
            logging.info(f"Processing new file: {source_file}")

        # 4. Embedding holen und speichern
        embedding = get_embedding(protocol_text)
        store_protocol(conn, protocol_text, embedding, protocol_data["metadata"])

        # Wartezeit von 2 Sekunden zwischen den API-Aufrufen, um Free-Tier-Limits zu respektieren.
        time.sleep(2) 

    if conn:
        conn.close()
        logging.info("=" * 40)
        logging.info("Indexing-Prozess abgeschlossen. Datenbankverbindung geschlossen.")

if __name__ == "__main__":
    main()
