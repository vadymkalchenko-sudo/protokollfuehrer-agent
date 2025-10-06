import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai
from pgvector.psycopg2 import register_vector
import psycopg2

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Environment Variable Loading ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Validation ---
if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    logging.error("Missing one or more required environment variables (SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY).")
    exit(1)

# --- Client Initialization ---
try:
    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logging.info("Successfully connected to Supabase.")

    # Configure Gemini API
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("Gemini API configured.")

    # Connect to PostgreSQL for pgvector
    # Supabase uses postgresql://postgres:[YOUR-PASSWORD]@[AWS-REGION].pooler.supabase.com:6543/postgres
    # We need to construct the DSN from the Supabase URL and Key (which often acts as the password for db access)
    db_url_parts = SUPABASE_URL.replace("https://", "").replace(".supabase.co", "").split('@')
    db_host_port = db_url_parts[0].split('.')
    db_host = f"{db_host_port[1]}.pooler.supabase.com"
    db_name = "postgres"
    db_user = "postgres"
    db_password = SUPABASE_KEY # Common practice for Supabase direct connections

    # Correct f-string usage, avoiding backslash issues.
    # The DSN format is standard and doesn't contain problematic backslashes.
    dsn = f"postgresql://{db_user}:{db_password}@{db_host}:6543/{db_name}"
    conn = psycopg2.connect(dsn)
    logging.info("Successfully connected to the database for pgvector registration.")
    register_vector(conn)

except Exception as e:
    logging.error(f"Failed to initialize clients: {e}")
    exit(1)


def get_embedding(text: str, model: str = "models/embedding-001") -> list[float]:
    """
    Generates a vector embedding for the given text using Gemini.
    """
    try:
        logging.info(f"Generating embedding for text snippet: '{text[:50]}...'")
        result = genai.embed_content(model=model, content=text)
        return result['embedding']
    except Exception as e:
        logging.error(f"Failed to generate embedding: {e}")
        return []

def store_protocol(text: str, embedding: list[float], metadata: dict) -> None:
    """
    Stores the protocol text, its embedding, and metadata in the Supabase 'protokolle' table.
    """
    if not embedding:
        logging.warning("Skipping storage due to empty embedding.")
        return

    try:
        logging.info(f"Storing protocol with metadata: {metadata}")
        data, count = supabase.table('protokolle').insert({
            'text': text,
            'embedding': embedding,
            'metadata': metadata
        }).execute()
        logging.info(f"Successfully stored protocol. Response: {data}")
    except Exception as e:
        # Log the detailed error from the database if possible
        logging.error(f"Failed to store protocol. Error: {e}")


def main():
    """
    Main function to process and index meeting protocols.
    """
    logging.info("Starting the indexing process...")

    # Sample meeting protocol data
    meeting_protocols = [
        {
            "text": "Meeting 1: Q3 Planning. Topics discussed: budget allocation, resource management, and setting new KPIs for the marketing team. Action items assigned to John and Maria.",
            "metadata": {"date": "2024-07-15", "meeting_id": "M1", "department": "Marketing"}
        },
        {
            "text": "Meeting 2: Technical Sprint Review. The engineering team presented the new features for the mobile app. A bug was identified in the payment gateway integration. The bug was assigned to the backend team.",
            "metadata": {"date": "2024-07-16", "meeting_id": "M2", "department": "Engineering"}
        },
        {
            "text": "Meeting 3: All-Hands Update. CEO shared the company's performance for H1 2024. New strategic partnerships were announced. An open Q&A session was held.",
            "metadata": {"date": "2024-07-18", "meeting_id": "M3", "department": "All"}
        }
    ]

    for protocol in meeting_protocols:
        protocol_text = protocol["text"]
        protocol_metadata = protocol["metadata"]

        # 1. Generate embedding
        embedding = get_embedding(protocol_text)

        # 2. Store in Supabase
        if embedding:
            store_protocol(protocol_text, embedding, protocol_metadata)
        else:
            logging.warning(f"Could not process protocol {protocol_metadata['meeting_id']} due to embedding failure.")

    logging.info("Indexing process completed.")
    if conn:
        conn.close()
        logging.info("Database connection closed.")


if __name__ == "__main__":
    main()