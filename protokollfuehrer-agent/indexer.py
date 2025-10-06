import os
import psycopg2
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
SUPABASE_URI = os.getenv("SUPABASE_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initial data
initial_rules = [
    "Die Zugangsdateien müssen für die zwei Arbeitsplätze GESONDERT behandelt werden, um Verbindungsprobleme zu vermeiden.",
    "Die Datenbankstruktur ist auf das Speichern von Daten als JSONB festgelegt.",
    "Das Team besteht aus vier Rollen (Architekt/Sie, Entwicklungsleiter/Gemini, Code-Entwickler/Jules, Gedächtnis/Protokollführer). Jules hat KEINEN Zugriff auf lokale Datenbanken.",
    "Wir kämpfen derzeit mit der Aktenverwaltung und können keine stabilen Daten aus dem physischen Server schreiben."
]

def setup_database(conn):
    """Sets up the database table and enables the vector extension if not present."""
    with conn.cursor() as cur:
        print("Checking for vector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("Vector extension checked/created.")

        print("Creating table public.protokoll_wissen if it doesn't exist...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.protokoll_wissen (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                embedding vector(768)
            );
        """)
        print("Table public.protokoll_wissen checked/created.")
        conn.commit()

def main():
    """Connects to Supabase, vectorizes initial data, and stores it."""
    if not SUPABASE_URI or not GEMINI_API_KEY:
        print("Error: SUPABASE_URI and GEMINI_API_KEY must be set in the .env file.")
        return

    try:
        # Configure the Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        model = 'models/text-embedding-004'
        print("Gemini API configured.")

        # Connect to Supabase
        print("Connecting to Supabase...")
        with psycopg2.connect(SUPABASE_URI) as conn:
            print("Database connection successful.")

            # Setup database schema
            setup_database(conn)

            with conn.cursor() as cur:
                # Clear existing data to avoid duplicates on re-run
                print("Clearing existing data from protokoll_wissen...")
                cur.execute("DELETE FROM public.protokoll_wissen;")
                print("Existing data cleared.")

                # Process and insert each rule
                for rule_text in initial_rules:
                    print(f"Embedding rule: '{rule_text[:30]}...'")
                    # Create the embedding
                    embedding = genai.embed_content(
                        model=model,
                        content=rule_text,
                        task_type="RETRIEVAL_DOCUMENT"
                    )["embedding"]

                    print("Embedding created. Inserting into database...")
                    # Insert data into the table
                    cur.execute(
                        "INSERT INTO public.protokoll_wissen (content, embedding) VALUES (%s, %s)",
                        (rule_text, embedding)
                    )
                    print("Insertion successful.")

                conn.commit()
                print("\nAll initial rules have been successfully vectorized and stored.")

    except psycopg2.OperationalError as e:
        print(f"Database connection error: {e}")
        print("Please ensure your SUPABASE_URI is correct and the database is accessible.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
    print("\nIndexer script finished.")