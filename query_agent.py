import os
import sys
import logging
from dotenv import load_dotenv
# KORREKTUR: Der Import muss den vollen Pfad verwenden
import google.generativeai as genai
from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2 import sql 
from urllib.parse import urlparse 
import time

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
# NEU: Verwendet den Supabase Dateinamen, der im Dockerfile verfügbar sein muss.
SUPABASE_CERT_PATH = os.path.join(os.path.dirname(__file__), 'prod-ca-2021.crt')

def initialize_clients():
    """Initialisiert Gemini API und Datenbankverbindung."""
    
    # --- Validation ---
    if not all([GEMINI_API_KEY, DB_CONNECTION_URI]):
        logging.error("🚨 FEHLER: Die kritischen Umgebungsvariablen GEMINI_API_KEY oder DB_CONNECTION_URI fehlen.")
        logging.error("HINWEIS: Bitte stellen Sie sicher, dass die .env-Datei korrekt befüllt ist.")
        sys.exit(1)
        
    # Prüfen, ob das Zertifikat existiert (muss vom lokalen System ins Docker-Image kopiert werden!)
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
        # Fängt jeden Fehler ab, der beim Verbinden auftritt (z.B. falsche URI, falsches Passwort, IP-Allowlist)
        logging.error(f"🚨 KRITISCHER FEHLER bei der Initialisierung oder Datenbankverbindung: {e}")
        logging.error("HINWEIS: Möglicherweise liegt der Fehler noch an der 'Address not in tenant allow_list' (siehe Supabase Console).")
        sys.exit(1)


# --- RAG Core Logik ---

def get_embedding(text: str, model: str = "text-embedding-004") -> list[float]:
    """Generiert ein Vektor-Embedding für eine Suchanfrage."""
    # Vereinfachte Version ohne Retry-Logik für den Query-Agent
    try:
        response = genai.embed_content( 
            model=model,
            content=text,
            task_type="RETRIEVAL_QUERY" # Wichtig: task_type für die Suche
        )
        return response['embedding']
    except Exception as e:
        logging.error(f"🚨 FEHLER beim Generieren des Query Embeddings: {e}")
        return []

def retrieve_and_generate(conn, query: str):
    """
    Führt die RAG-Logik aus: 
    1. Generiert Embedding für die Suchanfrage.
    2. Sucht die ähnlichsten Dokumente (Protokolle) in der Datenbank.
    3. Generiert eine Antwort mit Gemini basierend auf den gefundenen Dokumenten.
    """
    query_embedding = get_embedding(query)
    if not query_embedding:
        logging.error("Abbruch der Suche: Query Embedding konnte nicht generiert werden.")
        return "Entschuldigung, die Suchanfrage konnte aufgrund eines API-Fehlers nicht verarbeitet werden."

    try:
        # Vektorsuche (Ähnlichkeitsmaß: Vektor-Operator <->)
        with conn.cursor() as cur:
            # Begrenzt die Suche auf die 3 relevantesten Protokolle
            # <-> ist der L2-Abstands-Operator (kleiner ist besser/ähnlicher)
            cur.execute(
                """
                SELECT text, metadata
                FROM protokolle
                ORDER BY embedding <-> %s
                LIMIT 3
                """,
                # FIX: pgvector benötigt eckige Klammern []
                ('[' + ', '.join(map(str, query_embedding)) + ']',) 
            )
            results = cur.fetchall()
            
            if not results:
                logging.info("Keine relevanten Protokolle in der Datenbank gefunden.")
                return f"Ich konnte keine Protokolle finden, die sich auf '{query}' beziehen."
            
            # 3. Kontext vorbereiten
            context = "\n---\n".join([f"Protokoll ({r[1].get('meeting_id', 'N/A')}, {r[1].get('date', 'N/A')}): {r[0]}" for r in results])
            logging.info(f"✅ {len(results)} Dokumente für RAG gefunden und Kontext vorbereitet.")

            # 4. Antwort generieren
            system_instruction = (
                "Du bist ein Protokoll-Experte und KI-Assistent. Deine Aufgabe ist es, "
                "die Benutzerfrage präzise und freundlich zu beantworten, basierend ausschließlich auf dem bereitgestellten Kontext. "
                "Wenn die Antwort nicht im Kontext enthalten ist, sage höflich, dass du die Informationen nicht finden konntest."
            )
            
            prompt = (
                f"Antworte auf die Frage des Benutzers basierend auf dem folgenden Kontext:\n\n"
                f"KONTEXT:\n{context}\n\n"
                f"BENUTZERFRAGE: {query}"
            )
            
            model = genai.GenerativeModel(
                'gemini-2.5-flash', # FIX: Modellname als Positionsargument übergeben, nicht als Schlüsselwortargument 'model='
                system_instruction=system_instruction
            )
            
            response = model.generate_content(prompt)
            
            return response.text

    except Exception as e:
        logging.error(f"🚨 FEHLER bei der Datenbankabfrage oder Generierung: {e}")
        return "Ein interner Fehler ist bei der Datenverarbeitung aufgetreten."


def main():
    """
    Hauptfunktion des Query Agents zur Beantwortung von Fragen.
    """
    logging.info("=" * 40)
    logging.info("Protokoll Query Agent - Start")
    logging.info("=" * 40)
    
    conn = initialize_clients()
    
    # Beispiel-Abfrage
    query = "Welche Maßnahmen wurden im Q3-Planungsmeeting für das Marketingteam beschlossen?"
    
    logging.info(f"🔍 Starte RAG-Suche für Frage: '{query}'")
    
    # Warte kurz, falls der Indexer gerade läuft (damit die Daten da sind)
    time.sleep(5) 

    try:
        answer = retrieve_and_generate(conn, query)
        
        logging.info("=" * 40)
        logging.info(f"✨ ANTWORT DES AGENTEN ✨")
        logging.info(f"\n{answer}")
        logging.info("=" * 40)
    finally:
        if conn:
            conn.close()
            logging.info("Datenbankverbindung geschlossen.")

if __name__ == "__main__":
    main()
