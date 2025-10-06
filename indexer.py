import os
import logging
import json
from dotenv import load_dotenv
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
DB_CONNECTION_URI = os.getenv("DB_CONNECTION_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Validation ---
if not all([DB_CONNECTION_URI, GEMINI_API_KEY]):
    logging.error("Missing one or more required environment variables (DB_CONNECTION_URI, GEMINI_API_KEY).")
    exit(1)

# --- Client Initialization ---
conn = None
try:
    # Configure Gemini API
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("Gemini API configured.")

    # Connect to PostgreSQL using the direct Transaction Pooler URI
    logging.info("Connecting to the database via Transaction Pooler...")
    conn = psycopg2.connect(DB_CONNECTION_URI)
    register_vector(conn)
    logging.info("Successfully connected to the database and registered pgvector.")

except Exception as e:
    logging.error(f"Failed to initialize clients or connect to the database: {e}")
    if conn:
        conn.close()
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

def store_protocol(conn, text: str, embedding: list[float], metadata: dict) -> None:
    """
    Stores the protocol text, its embedding, and metadata in the 'protokolle' table using psycopg2.
    """
    if not embedding:
        logging.warning("Skipping storage due to empty embedding.")
        return

    # Convert metadata dict to a JSON string for the JSONB column
    metadata_json = json.dumps(metadata)

    try:
        logging.info(f"Storing protocol with metadata: {metadata}")
        # Use a `with` statement for the cursor to ensure it's closed automatically
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO protokolle (text, embedding, metadata) VALUES (%s, %s, %s)",
                (text, embedding, metadata_json)
            )
        # Commit the transaction to make the changes permanent
        conn.commit()
        logging.info(f"Successfully stored protocol for meeting_id: {metadata.get('meeting_id')}")
    except Exception as e:
        logging.error(f"Failed to store protocol. Error: {e}")
        # Rollback the transaction in case of an error
        conn.rollback()


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
            store_protocol(conn, protocol_text, embedding, protocol_metadata)
        else:
            logging.warning(f"Could not process protocol {protocol_metadata['meeting_id']} due to embedding failure.")

    logging.info("Indexing process completed.")
    if conn:
        conn.close()
        logging.info("Database connection closed.")


if __name__ == "__main__":
    main()