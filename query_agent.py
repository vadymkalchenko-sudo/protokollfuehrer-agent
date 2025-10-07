import os
import logging
import json
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from google import genai
from google.genai.errors import APIError
from time import sleep

# --- Konfiguration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Secrets aus .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_CONNECTION_URI = os.getenv("DB_CONNECTION_URI")

# Modell- und Suchparameter
EMBEDDING_MODEL = "text-embedding-004"
RAG_MODEL = "gemini-2.5-flash" 
EXPECTED_DIMENSIONS = 768
TOP_K_RESULTS = 2 # Anzahl der relevantesten Protokolle, die gesucht werden

# --- Datenbank und API Initialisierung ---

def initialize_client_and_db():
    """Initialisiert den Gemini Client und die Datenbankverbindung."""
    try:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY ist nicht in der .env-Datei gesetzt.")
        
        # Initialisiere Gemini Client
        client = genai.Client(api_key=GEMINI_API_KEY)
        logging.info("✅ Gemini API erfolgreich konfiguriert.")
    except Exception as e:
        logging.error(f"🚨 FEHLER beim Initialisieren des Gemini Clients: {e}")
        return None, None

    try:
        # PostgreSQL/Supabase Verbindung herstellen
        # Nutzt das lokal behaltene Zertifikat zur SSL-Verifizierung
        conn = psycopg2.connect(DB_CONNECTION_URI, sslmode='verify-full', sslrootcert='prod-ca-2021.crt')
        conn.autocommit = True 
        logging.info("✅ Datenbankverbindung über Pooler URI mit SSL/Zertifikatprüfung hergestellt.")
    except psycopg2.Error as e:
        logging.error(f"🚨 FEHLER bei der Datenbankverbindung: {e}")
        logging.warning("HINWEIS: Prüfen Sie die DB_CONNECTION_URI und das 'prod-ca-2021.crt' Zertifikat.")
        return client, None
    
    return client, conn

# --- Core-Funktionen ---

def generate_embedding(client: genai.Client, text: str):
    """Generiert ein Vektor-Embedding für den gegebenen Text."""
    try:
        # Hier generieren wir das Query-Embedding
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_QUERY", # WICHTIG: Task-Type ist Query für die Suche
            title="User Query for Protocol Search"
        )
        embedding = response['embedding']
        
        if len(embedding) != EXPECTED_DIMENSIONS:
             logging.warning(f"WARNUNG: Erwartete Dimensionen {EXPECTED_DIMENSIONS}, erhalten: {len(embedding)}")

        return embedding
    except APIError as e:
        logging.error(f"🚨 FEHLER bei der Gemini API (Embedding): {e}")
        return None
    except Exception as e:
        logging.error(f"🚨 Unbekannter Fehler beim Embedding: {e}")
        return None

def fetch_relevant_protocols(conn, query_embedding: list):
    """Führt die Vektor-Ähnlichkeitssuche (KNN) durch."""
    cursor = conn.cursor()
    
    # Konvertiere Embedding in das pgvector-Format (eckige Klammern [])
    query_embedding_str = '[' + ', '.join(map(str, query_embedding)) + ']'
    
    try:
        # Sucht nach den TOP_K_RESULTS ähnlichsten Protokollen in der Datenbank
        # <-> ist der pgvector Entfernungsoperator (Euklidische Distanz)
        query = sql.SQL("""
            SELECT 
                text, 
                metadata->>'meeting_id' AS meeting_id, 
                1 - (embedding <-> %s) AS similarity_score
            FROM 
                protokolle
            ORDER BY 
                embedding <-> %s
            LIMIT %s;
        """)
        
        cursor.execute(query, [query_embedding_str, query_embedding_str, TOP_K_RESULTS])
        results = cursor.fetchall()
        
        relevant_protocols = []
        for text, meeting_id, score in results:
            relevant_protocols.append({
                "meeting_id": meeting_id,
                "text": text,
                "similarity": score
            })
            logging.info(f"   -> Gefunden: {meeting_id} (Ähnlichkeit: {score:.4f})")

        return relevant_protocols
    except psycopg2.Error as e:
        logging.error(f"🚨 FEHLER bei der Vektor-Suche in der Datenbank: {e}")
        return []
    finally:
        cursor.close()

def generate_rag_answer(client: genai.Client, query: str, protocols: list):
    """Generiert die Antwort basierend auf der Abfrage und dem Kontext (RAG)."""
    
    context_text = "\n\n--- Kontext Protokolle ---\n"
    for p in protocols:
        context_text += f"[Protokoll ID: {p['meeting_id']}]\n{p['text']}\n"
    
    # System Instruction für den LLM-Prompt
    system_instruction = (
        "Du bist ein Protokoll-Assistent. Deine Aufgabe ist es, die Benutzeranfrage ausschließlich "
        "basierend auf dem unten bereitgestellten 'Kontext Protokolle' zu beantworten. "
        "Antworte präzise, klar und gib immer die 'Protokoll ID' an, aus der du die Information entnommen hast. "
        "Wenn die Antwort im Kontext fehlt, antworte freundlich, dass du die Information nicht finden konntest."
    )
    
    full_prompt = (
        f"{context_text}\n\n--- Benutzerfrage ---\n"
        f"Frage: {query}\n\n"
        "Antwort: "
    )

    logging.info(f"-> Sende {len(protocols)} Protokolle an Gemini zur Beantwortung.")

    try:
        # Hier verwenden wir den "gemini-2.5-flash" für die schnelle RAG-Antwort
        response = client.models.generate_content(
            model=RAG_MODEL,
            contents=[full_prompt],
            system_instruction=system_instruction
        )
        return response.text
    except APIError as e:
        logging.error(f"🚨 FEHLER bei der Gemini API (RAG): {e}")
        return "Entschuldigung, beim Generieren der Antwort ist ein API-Fehler aufgetreten."

# --- Hauptlogik ---

def run_query_agent():
    """Führt den gesamten Abfrage-Prozess durch."""
    client, conn = initialize_client_and_db()

    if not client or not conn:
        return

    # Beispiel-Abfrage (Hardcoded für den Docker-Test)
    user_query = "Welche Action Items wurden an John oder Maria vergeben und worum ging es dabei?"
    logging.info("========================================")
    logging.info(f"⭐ Benutzeranfrage: {user_query}")
    logging.info("========================================")

    # 1. Embedding für die Abfrage generieren
    query_embedding = generate_embedding(client, user_query)

    if query_embedding:
        # 2. Relevante Protokolle via Vektor-Suche finden
        logging.info("-> Starte Vektor-Suche nach relevanten Protokollen...")
        relevant_protocols = fetch_relevant_protocols(conn, query_embedding)

        if relevant_protocols:
            # 3. RAG-Antwort generieren
            final_answer = generate_rag_answer(client, user_query, relevant_protocols)
            
            logging.info("========================================")
            logging.info("✅ Generierte Antwort (RAG):")
            print(final_answer)
            logging.info("========================================")
        else:
            logging.warning("Keine relevanten Protokolle gefunden, die zur Beantwortung genutzt werden können.")
    
    if conn:
        conn.close()

if __name__ == "__main__":
    run_query_agent()
