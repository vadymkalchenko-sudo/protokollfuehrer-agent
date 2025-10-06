import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import psycopg2
from psycopg2 import sql

# --- Basic Configuration ---
# Set up logging to provide clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
# This loads variables from the .env file. In the Docker environment,
# docker-compose will pass these variables from the .env file.
load_dotenv()

# --- Gemini API Initialization ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or "your_gemini_api_key_here" in GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY environment variable not found or is a placeholder. Please set it in your .env file.")
    exit(1)
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("Gemini API configured successfully.")
except Exception as e:
    logging.error(f"Failed to configure Gemini API: {e}")
    exit(1)

# --- Supabase/PostgreSQL Database Configuration ---
SUPABASE_URI = os.getenv("SUPABASE_URI")
if not SUPABASE_URI or "your_supabase_uri_here" in SUPABASE_URI:
    logging.error("SUPABASE_URI environment variable not found or is a placeholder. Please set it in your .env file.")
    exit(1)

def get_db_connection():
    """Establishes a connection to the PostgreSQL database using the connection URI."""
    try:
        conn = psycopg2.connect(SUPABASE_URI)
        logging.info("Database connection established successfully.")
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Could not connect to the database. Please check your SUPABASE_URI. Error: {e}")
        return None

def create_embedding(text, model="text-embedding-004"):
    """
    Generates a vector embedding for the given text using the specified Gemini model.
    """
    if not text or not isinstance(text, str):
        logging.warning("Input for embedding must be a non-empty string.")
        return None
    try:
        logging.info(f"Generating embedding for text snippet: '{text[:60].replace('\n', ' ')}...'")
        result = genai.embed_content(model=f"models/{model}", content=text)
        logging.info("Embedding generated successfully.")
        return result['embedding']
    except Exception as e:
        logging.error(f"An error occurred during embedding generation: {e}")
        return None

def store_protocol_entry(text, vector_embedding, metadata):
    """
    Stores the protocol text, its vector embedding, and metadata in the 'protokolle' table.
    """
    if not vector_embedding:
        logging.error("Cannot store entry without a vector embedding.")
        return False

    # SQL query to insert data into the 'protokolle' table.
    # The table is expected to have columns: text (TEXT), vector_embedding (VECTOR), and metadata (JSONB).
    insert_query = sql.SQL("INSERT INTO protokolle (text, vector_embedding, metadata) VALUES (%s, %s, %s)")

    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # pgvector expects the vector as a string representation like '[1,2,3,...]'
                embedding_str = str(vector_embedding)
                # The metadata column is expected to be JSONB, so we dump the dict to a JSON string.
                metadata_json = json.dumps(metadata)

                cur.execute(insert_query, (text, embedding_str, metadata_json))
                conn.commit()
                logging.info("Successfully inserted protocol entry into the database.")
                return True
        except Exception as e:
            logging.error(f"Failed to insert data into the database: {e}")
            conn.rollback()  # Roll back the transaction on error
            return False
        finally:
            conn.close()
            logging.info("Database connection closed.")
    return False

def main():
    """
    Main function to execute the end-to-end indexing process.
    """
    logging.info("--- Starting Protokoll Indexing Agent ---")

    # This is example data. In a real-world application, you would read this
    # from a file, a message queue, or an API.
    protocol_text = (
        "Protokoll: Kick-off Meeting 'Protokollführer-Agent'\n"
        "Datum: 06.10.2025\n"
        "Teilnehmer: Jules, Entwicklerteam\n"
        "Thema: Die Hauptanforderung ist die Entwicklung eines robusten Indexing-Agenten. "
        "Die Docker-Konfiguration muss den Anwendungscode direkt in das Image kopieren, "
        "um 'ModuleNotFoundError' zu vermeiden. Der Agent nutzt Gemini für Embeddings "
        "und speichert diese in einer Supabase PostgreSQL-Datenbank."
    )
    protocol_metadata = {
        "source": "internal_meeting_notes",
        "project_id": "protokollfuehrer-agent-v2",
        "status": "final"
    }

    # 1. Generate the vector embedding for the protocol text
    embedding = create_embedding(protocol_text)

    # 2. If embedding was created successfully, store the data in the database
    if embedding:
        success = store_protocol_entry(protocol_text, embedding, protocol_metadata)
        if success:
            logging.info("--- Indexing process completed successfully. ---")
        else:
            logging.error("--- Indexing process failed during database operation. ---")
    else:
        logging.error("--- Indexing process failed: Could not generate embedding. ---")

if __name__ == "__main__":
    # This script is designed to be run once and then exit.
    # The Docker container will execute this script and then stop.
    main()