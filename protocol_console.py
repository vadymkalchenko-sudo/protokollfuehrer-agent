import asyncio
import logging
import os
import sys
from typing import List, Optional

import asyncpg
import google.generativeai as genai
from colorama import Fore, Style, init
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector
import json

# --- Initialization ---
init(autoreset=True)
# Load environment variables
# 1. Lade die öffentliche Vorlage (.env)
load_dotenv(".env")
# 2. Lade die lokale, private Schlüssel-Datei (.env.local).
#    Diese MUSS die Werte in der .env ueberschreiben.
load_dotenv(".env.local", override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Variable Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_CONNECTION_URI = os.getenv("DB_CONNECTION_URI")

def check_env_variables():
    """Checks for necessary environment variables and exits if they are missing."""
    if not GEMINI_API_KEY or not DB_CONNECTION_URI:
        logging.error(f"{Fore.RED}🚨 Critical environment variables (DB_CONNECTION_URI or GEMINI_API_KEY) are missing. Please create a .env file or set them. Program terminated.")
        sys.exit(1)
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info(f"{Fore.GREEN}Environment variables loaded successfully.")

# --- Database Operations ---
async def get_db_connection() -> asyncpg.Connection:
    """Establishes a database connection using the connection URI."""
    try:
        conn = await asyncpg.connect(DB_CONNECTION_URI)
        await register_vector(conn)
        logging.info(f"{Fore.GREEN}Database connection established successfully.")
        return conn
    except Exception as e:
        logging.error(f"{Fore.RED}🚨 Could not connect to the database: {e}")
        sys.exit(1)

async def ensure_schema(conn: asyncpg.Connection):
    """Ensures the pgvector extension is enabled and the 'protokolle' table exists."""
    try:
        logging.info(f"{Fore.CYAN}Ensuring database schema exists...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS protokolle (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding VECTOR(768),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        logging.info(f"{Fore.GREEN}Database schema verified and ready.")
    except Exception as e:
        logging.error(f"{Fore.RED}🚨 Error ensuring database schema: {e}")
        sys.exit(1)

async def embed_and_store_text(conn: asyncpg.Connection, text: str, source: str) -> bool:
    """Generates embedding for text and stores it in the database."""
    logging.info(f"{Fore.CYAN}Generating embedding for source: {source}...")
    try:
        embedding_model = "text-embedding-004"
        embedding_response = await asyncio.to_thread(
            genai.embed_content,
            model=embedding_model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embedding = embedding_response['embedding']
        metadata = {'source': source}

        # KRITISCHER FIX: Konvertiert das Python Dictionary in einen JSON-String,
        # da die Datenbank (jsonb) einen String erwartet.
        metadata_json = json.dumps(metadata)

        logging.info(f"{Fore.CYAN}Storing text and embedding in the database...")
        # The `register_vector` call for asyncpg handles the list-to-vector conversion,
        # so explicit string formatting of the embedding is not needed.
        await conn.execute(
            "INSERT INTO protokolle (text, embedding, metadata) VALUES ($1, $2, $3)",
            text,
            embedding,
            metadata_json
        )
        logging.info(f"{Fore.GREEN}Successfully stored the manifest from source: {source}")
        return True
    except Exception as e:
        logging.error(f"{Fore.RED}🚨 An error occurred during embedding or storage: {e}")
        return False

# --- RAG Query Operations ---
async def handle_query_input(conn: asyncpg.Connection):
    """Handles the entire RAG cycle from user input to final answer."""
    # 1. Benutzer-Eingabe
    question = input(Fore.WHITE + "Stellen Sie Ihre Frage: ")
    if not question.strip():
        logging.warning("Leere Frage eingegeben. Vorgang abgebrochen.")
        return

    try:
        # 2. Query-Embedding
        logging.info("Generiere Vektor-Embedding für die Frage...")
        embedding_model = "text-embedding-004"
        embedding_response = await asyncio.to_thread(
            genai.embed_content,
            model=embedding_model,
            content=question,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = embedding_response['embedding']

        # 3. Datenbank-Retrieval (Vektor-Suche)
        logging.info("Suche nach relevanten Dokumenten in der Datenbank...")
        # Die Cosinus-Distanz ist 1 - Cosinus-Ähnlichkeit. Wir wollen die Ähnlichkeit.
        # Top-K=3
        retrieved_records = await conn.fetch(
            """
            SELECT
                text,
                metadata->>'source' AS source,
                1 - (embedding <=> $1) AS similarity
            FROM protokolle
            ORDER BY embedding <=> $1
            LIMIT 3
            """,
            query_embedding
        )

        if not retrieved_records:
            logging.warning("Keine relevanten Dokumente im Vektor-Index gefunden.")
            print(Fore.YELLOW + "Ich konnte keine relevanten Informationen zu Ihrer Frage finden.")
            return

        # 4. Kontext-Erstellung
        logging.info(f"{len(retrieved_records)} relevante Dokumente gefunden. Erstelle Kontext...")
        context_parts = []
        for rec in retrieved_records:
            source = rec['source'] or 'Unbekannt'
            similarity = rec['similarity']
            text = rec['text']
            context_part = (
                f"Quelle: {source}, Ähnlichkeit: {similarity:.4f}\n"
                f"---\n"
                f"{text}"
            )
            context_parts.append(context_part)

        context = "\n\n".join(context_parts)

        # 5. Gemini-Generierung
        logging.info("Generiere finale Antwort mit dem Gemini-Modell...")
        model = genai.GenerativeModel('gemini-2.5-flash')

        system_prompt = (
            "Du bist ein spezialisierter Assistent für Protokoll-Analyse. "
            "Deine Aufgabe ist es, die folgende Frage prägnant und ausschließlich basierend auf dem bereitgestellten Kontext zu beantworten. "
            "Fasse die relevanten Informationen zusammen. Gib ehrlich zu, wenn die Antwort nicht im Kontext enthalten ist. "
            "Verwende keine externen Wissensquellen."
        )

        final_prompt = f"{system_prompt}\n\nKONTEXT:\n{context}\n\nFRAGE:\n{question}"

        response = await asyncio.to_thread(
            model.generate_content,
            final_prompt
        )

        # 6. Output
        print(Fore.GREEN + "\n=== ANTWORT ===")
        print(Style.BRIGHT + response.text)
        print(Fore.GREEN + "===============\n")

    except Exception as e:
        # 6. Fehlerbehandlung
        logging.error(f"Ein Fehler ist während der Abfrageverarbeitung aufgetreten: {e}", exc_info=True)
        # asyncpg's connection objects don't have a rollback method.
        # Transactions are managed via Transaction objects. For a single query, this is not needed.
        # We'll log the error and inform the user.
        print(Fore.RED + "Es ist ein Fehler aufgetreten. Bitte überprüfen Sie die Logs.")

# --- Main Application Loop ---
def print_menu():
    """Prints the interactive menu."""
    print(Fore.CYAN + "\n--- Protokollführer CLI ---")
    print(Fore.YELLOW + "1. Manifest einspeisen (Text eingeben)")
    print(Fore.YELLOW + "2. Frage stellen (RAG-Abfrage)")
    print(Fore.YELLOW + "3. Beenden")
    print(Style.RESET_ALL)

async def main():
    """The main function to run the CLI."""
    check_env_variables()
    conn = await get_db_connection()
    await ensure_schema(conn)

    try:
        while True:
            print_menu()
            choice = input(Fore.WHITE + "Wähle eine Option: ")

            if choice == '1':
                print(Fore.CYAN + "Geben Sie den Text des Manifests ein. Geben Sie 'ENDE' in eine neue Zeile ein, um zu beenden.")
                manifest_text = []
                while True:
                    try:
                        line = input()
                        if line.strip().upper() == 'ENDE':
                            break
                        manifest_text.append(line)
                    except EOFError:
                        break

                if manifest_text:
                    full_text = "\n".join(manifest_text)
                    source_name = input(Fore.WHITE + "Geben Sie einen Quellnamen für dieses Manifest an (z.B. 'Meeting-2024-10-08'): ")
                    if not source_name:
                        source_name = "manual_input"
                    await embed_and_store_text(conn, full_text, source_name)
                else:
                    logging.warning(f"{Fore.YELLOW}Kein Text eingegeben.")

            elif choice == '2':
                await handle_query_input(conn)

            elif choice == '3':
                print(Fore.GREEN + "Anwendung wird beendet. Auf Wiedersehen!")
                break

            else:
                logging.warning(f"{Fore.RED}Ungültige Auswahl. Bitte versuchen Sie es erneut.")
    finally:
        await conn.close()
        logging.info(f"{Fore.GREEN}Database connection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(Fore.GREEN + "\nAnwendung wird durch Benutzer beendet.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"{Fore.RED}\nEin unerwarteter Fehler ist aufgetreten: {e}")
        sys.exit(1)