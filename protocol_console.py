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

        logging.info(f"{Fore.CYAN}Storing text and embedding in the database...")
        await conn.execute(
            "INSERT INTO protokolle (text, embedding, metadata) VALUES ($1, $2, $3)",
            text,
            embedding,
            {'source': source}
        )
        logging.info(f"{Fore.GREEN}Successfully stored the manifest from source: {source}")
        return True
    except Exception as e:
        logging.error(f"{Fore.RED}🚨 An error occurred during embedding or storage: {e}")
        return False

# --- RAG Query Operations ---
async def find_similar_texts(conn: asyncpg.Connection, question: str) -> Optional[str]:
    """Finds texts in the database similar to the user's question."""
    logging.info(f"{Fore.CYAN}Generating embedding for the question...")
    try:
        embedding_model = "text-embedding-004"
        embedding_response = await asyncio.to_thread(
            genai.embed_content,
            model=embedding_model,
            content=question,
            task_type="RETRIEVAL_QUERY"
        )
        embedding = embedding_response['embedding']

        logging.info(f"{Fore.CYAN}Searching for relevant context in the database...")
        # Using cosine distance (1 - cosine_similarity) with the <=> operator
        records = await conn.fetch(
            "SELECT text FROM protokolle ORDER BY embedding <=> $1 LIMIT 5",
            embedding
        )

        if not records:
            logging.warning(f"{Fore.YELLOW}No relevant information found in the database.")
            return None

        context = "\n---\n".join([rec['text'] for rec in records])
        return context
    except Exception as e:
        logging.error(f"{Fore.RED}🚨 An error occurred during similarity search: {e}")
        return None

async def ask_question_with_context(question: str, context: str) -> None:
    """Asks the generative model a question based on the provided context."""
    logging.info(f"{Fore.CYAN}Asking the Gemini model for an answer...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-preview-0514')

        system_instruction = (
            "You are a helpful assistant. Your name is 'Protokollführer-Agent'. "
            "Answer the user's question based *only* on the provided context. "
            "If the answer is not found in the context, say 'I cannot answer this question based on the provided information.' "
            "Do not use any external knowledge. Be concise and precise."
        )

        # The 'system_instruction' parameter is not directly supported in all versions/methods.
        # Prepending it to the user prompt is a common and reliable workaround.
        prompt = f"{system_instruction}\n\nContext:\n{context}\n\nQuestion: {question}"

        response = await asyncio.to_thread(
            model.generate_content,
            prompt
        )

        print(Fore.MAGENTA + "\n--- Antwort des Agenten ---")
        print(Style.BRIGHT + response.text)
        print(Fore.MAGENTA + "--------------------------\n")

    except Exception as e:
        logging.error(f"{Fore.RED}🚨 An error occurred while communicating with the Gemini API: {e}")

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
                question = input(Fore.WHITE + "Stellen Sie Ihre Frage: ")
                if not question:
                    logging.warning(f"{Fore.YELLOW}Keine Frage eingegeben.")
                    continue

                context = await find_similar_texts(conn, question)
                if context:
                    await ask_question_with_context(question, context)

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