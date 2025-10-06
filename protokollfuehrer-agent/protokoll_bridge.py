import os
import sys
import psycopg2
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
SUPABASE_URI = os.getenv("SUPABASE_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def search_memory(query_string: str):
    """
    Searches the Supabase memory for the top 3 most similar text chunks.
    """
    if not query_string:
        print("Error: No search string provided.")
        return

    if not SUPABASE_URI or not GEMINI_API_KEY:
        print("Error: SUPABASE_URI and GEMINI_API_KEY must be set in the .env file.")
        return

    try:
        # Configure the Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        model = 'models/text-embedding-004'

        # Vectorize the search query
        print(f"Vectorizing query: '{query_string}'")
        query_embedding = genai.embed_content(
            model=model,
            content=query_string,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]
        print("Query vectorized.")

        # Connect to Supabase
        print("Connecting to database...")
        with psycopg2.connect(SUPABASE_URI) as conn:
            print("Database connection successful. Performing search...")
            with conn.cursor() as cur:
                # Perform the vector similarity search
                cur.execute(
                    "SELECT id, created_at, content FROM public.protokoll_wissen ORDER BY embedding <=> %s LIMIT 3",
                    (str(query_embedding),)
                )
                results = cur.fetchall()
                print("Search complete.")

        # Format and print the results
        print("\n### KONTEXT AUS PROTOKOLLFÜHRER (RAG-ABRUF):")
        if not results:
            print("- No relevant context found.")
        else:
            for row in results:
                doc_id, timestamp, content = row
                # Format timestamp to YYYY-MM-DD
                formatted_date = timestamp.strftime('%Y-%m-%d')
                print(f"- [Datum: {formatted_date}] [ID: {doc_id}] {content}")

    except psycopg2.OperationalError as e:
        print(f"Database connection error: {e}")
        print("Please ensure your SUPABASE_URI is correct and the database is accessible.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Main function to handle command-line arguments.
    """
    if len(sys.argv) < 2:
        print("Usage: python protokoll_bridge.py \"<your search query>\"")
        sys.exit(1)

    # Join all arguments after the script name to form the query
    search_query = " ".join(sys.argv[1:])
    search_memory(search_query)

if __name__ == "__main__":
    main()
    print("\nProtokoll-Bridge script finished.")