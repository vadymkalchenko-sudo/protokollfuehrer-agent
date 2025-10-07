import os
import logging
import sys
import json
import time # Für die Wartefunktion
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

def is_already_indexed(conn, source_file: str) -> bool:
    """
    Prüft, ob ein Protokoll mit dem gegebenen Dateinamen bereits in der DB indiziert ist.
    """
    try:
        with conn.cursor() as cur:
            # Effiziente Abfrage, die prüft, ob der 'source_file' Key im JSONB-Feld existiert und dem Wert entspricht.
            cur.execute(
                "SELECT 1 FROM protokolle WHERE metadata->>'source_file' = %s LIMIT 1",
                (source_file,)
            )
            # fetchone() gibt ein Tupel zurück, wenn ein Datensatz gefunden wird, sonst None.
            return cur.fetchone() is not None
    except Exception as e:
        logging.error(f"🚨 FEHLER bei der Duplikatsprüfung für '{source_file}': {e}")
        # Im Fehlerfall gehen wir vorsichtshalber davon aus, dass es nicht indiziert ist,
        # um Datenverlust zu vermeiden, aber loggen den Fehler deutlich.
        return False

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
    Hauptfunktion zur Verarbeitung und Indexierung von Besprechungsprotokollen.
    """
    logging.info("=" * 40)
    logging.info("Protokoll Indexing Agent - Start")
    logging.info("=" * 40)
    
    # Initialisiert Clients und bricht bei Fehler ab
    conn = initialize_clients()
    
    # Beispiel-Daten (angepasst, um 'source_file' zu enthalten)
    # In einer echten Implementierung würde dies aus dem Dateisystem geladen.
    meeting_protocols = [
        {
            "text": "Meeting 1: Q3 Planning. Topics discussed: budget allocation, resource management, and setting new KPIs for the marketing team. Action items assigned to John and Maria.",
            "metadata": {"date": "2024-07-15", "meeting_id": "M1", "department": "Marketing", "source_file": "protocols/meeting_q3_planning.txt"}
        },
        {
            "text": "Meeting 2: Technical Sprint Review. The engineering team presented the new features for the mobile app. A bug was identified in the payment gateway integration. The bug was assigned to the backend team.",
            "metadata": {"date": "2024-07-16", "meeting_id": "M2", "department": "Engineering", "source_file": "protocols/sprint_review_tech.md"}
        },
        {
            "text": "Meeting 3: All-Hands Update. CEO shared the company's performance for H1 2024. New strategic partnerships were announced. An open Q&A session was held.",
            "metadata": {"date": "2024-07-18", "meeting_id": "M3", "department": "All", "source_file": "protocols/all_hands_h1_2024.txt"}
        }
    ]

    for protocol in meeting_protocols:
        # Extrahiere den Dateinamen für die Duplikatsprüfung
        source_file = protocol["metadata"].get("source_file")

        if not source_file:
            logging.warning(f"Skipping protocol due to missing 'source_file' in metadata: {protocol['metadata']}")
            continue

        # Prüfen, ob die Datei bereits indiziert wurde
        if is_already_indexed(conn, source_file):
            logging.info(f"Skipping {source_file}: Already indexed.")
            continue # Nächste Datei verarbeiten

        # Nur wenn nicht bereits indiziert, Embedding holen und speichern
        logging.info(f"Processing new file: {source_file}")
        embedding = get_embedding(protocol["text"]) 
        store_protocol(conn, protocol["text"], embedding, protocol["metadata"])

        # Wartezeit von 2 Sekunden zwischen den API-Aufrufen, um Free-Tier-Limits zu respektieren.
        time.sleep(2) 

    if conn:
        conn.close()
        logging.info("=" * 40)
        logging.info("Indexing-Prozess abgeschlossen. Datenbankverbindung geschlossen.")

if __name__ == "__main__":
    main()
