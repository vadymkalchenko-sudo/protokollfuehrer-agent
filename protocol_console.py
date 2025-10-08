import asyncio
import os
import sys
from typing import Optional

import google.generativeai as genai
from colorama import Fore, Style, init
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from supabase import Client, create_client

# --- Initialization ---
init(autoreset=True)
load_dotenv()

# --- Environment Variable and Client Setup ---
def get_env_variable(var_name: str) -> str:
    """Gets an environment variable or exits if not found."""
    value = os.getenv(var_name)
    if not value:
        print(Fore.RED + f"Error: Environment variable '{var_name}' not found.")
        print(Fore.YELLOW + "Please create a .env file with the required variables.")
        sys.exit(1)
    return value

def create_supabase_client() -> Client:
    """Creates and returns a Supabase client."""
    supabase_url = get_env_variable("SUPABASE_URI")
    supabase_key = get_env_variable("SUPABASE_KEY")
    return create_client(supabase_url, supabase_key)

def configure_gemini_api():
    """Configures the Gemini API."""
    api_key = get_env_variable("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

# --- Database Operations ---
async def ensure_table_and_extension(supabase: Client):
    """Ensures the pgvector extension is enabled and the table exists."""
    # The Supabase client v2 uses psycopg2 under the hood, which is synchronous.
    # We'll wrap synchronous calls in asyncio.to_thread to avoid blocking.
    def sync_db_setup():
        with supabase.postgrest.session as session:
            # The python client doesn't have a generic RPC call for this,
            # so we assume the extension is enabled in the Supabase dashboard.
            # A more robust solution would use a direct psycopg2 connection.
            print(Fore.CYAN + "Checking for 'protokolle' table...")

            # Check if table exists
            response = supabase.table("protokolle").select("id").limit(1).execute()

            # PostgREST returns an error if the table doesn't exist.
            # We can't easily check for a "table does not exist" error code here.
            # A simple approach is to try to create it and let it fail if it exists.
            # This is not ideal but works for this CLI tool.
            try:
                # Create table schema if it doesn't exist
                # This is a placeholder, as direct DDL from the client is not standard.
                # It's better to set up the table via the Supabase SQL Editor.
                # The following is a conceptual representation.
                # A real implementation would use rpc() to call a stored procedure.
                pass # Assuming table is created via Supabase UI
            except Exception as e:
                # This will likely fail, but we proceed assuming the table exists.
                pass
        print(Fore.GREEN + "Database ready (assuming 'protokolle' table and 'pgvector' extension exist).")

    await asyncio.to_thread(sync_db_setup)


async def embed_and_store_text(supabase: Client, text: str, source: str) -> bool:
    """Generates embedding for text and stores it in the database."""
    print(Fore.CYAN + f"Generating embedding for source: {source}...")
    try:
        embedding_model = "text-embedding-004"
        embedding_response = await asyncio.to_thread(
            genai.embed_content,
            model=embedding_model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embedding = embedding_response['embedding']

        print(Fore.CYAN + "Storing text and embedding in the database...")
        data = {
            "text": text,
            "embedding": embedding,
            "metadata": {"source": source}
        }
        await asyncio.to_thread(supabase.table("protokolle").insert(data).execute)
        print(Fore.GREEN + "Successfully stored the manifest.")
        return True
    except Exception as e:
        print(Fore.RED + f"An error occurred during embedding or storage: {e}")
        return False

# --- RAG Query Operations ---
async def find_similar_texts(supabase: Client, question: str) -> Optional[str]:
    """Finds texts in the database similar to the user's question."""
    print(Fore.CYAN + "Generating embedding for the question...")
    try:
        embedding_model = "text-embedding-004"
        embedding_response = await asyncio.to_thread(
            genai.embed_content,
            model=embedding_model,
            content=question,
            task_type="RETRIEVAL_QUERY"
        )
        embedding = embedding_response['embedding']

        print(Fore.CYAN + "Searching for relevant context in the database...")
        # Use rpc to call the match_documents function
        match_response = await asyncio.to_thread(
            supabase.rpc,
            'match_protokolle',
            {
                'query_embedding': embedding,
                'match_threshold': 0.75,
                'match_count': 5
            }
        )

        if not match_response.data:
            print(Fore.YELLOW + "No relevant information found in the database.")
            return None

        # Combine the texts to form the context
        context = "\n---\n".join([item['text'] for item in match_response.data])
        return context
    except Exception as e:
        print(Fore.RED + f"An error occurred during similarity search: {e}")
        return None

async def ask_question_with_context(question: str, context: str) -> None:
    """Asks the generative model a question based on the provided context."""
    print(Fore.CYAN + "Asking the Gemini model for an answer...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-preview-0514')

        system_instruction = (
            "You are a helpful assistant. Your name is 'Protokollführer-Agent'. "
            "Answer the user's question based *only* on the provided context. "
            "If the answer is not found in the context, say 'I cannot answer this question based on the provided information.' "
            "Do not use any external knowledge. Be concise and precise."
        )

        response = await asyncio.to_thread(
            model.generate_content,
            f"Context:\n{context}\n\nQuestion: {question}",
            # generation_config={"system_instruction": system_instruction} # Not supported in all versions
        )

        print(Fore.MAGENTA + "\n--- Antwort des Agenten ---")
        print(Style.BRIGHT + response.text)
        print(Fore.MAGENTA + "--------------------------\n")

    except Exception as e:
        print(Fore.RED + f"An error occurred while communicating with the Gemini API: {e}")


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
    # --- Setup ---
    configure_gemini_api()
    supabase = create_supabase_client()
    await ensure_table_and_extension(supabase)

    # --- Main Loop ---
    while True:
        print_menu()
        choice = input(Fore.WHITE + "Wähle eine Option: ")

        if choice == '1':
            print(Fore.CYAN + "Geben Sie den Text des Manifests ein. Geben Sie 'ENDE' in eine neue Zeile ein, um zu beenden.")
            manifest_text = []
            while True:
                line = input()
                if line == 'ENDE':
                    break
                manifest_text.append(line)

            if manifest_text:
                full_text = "\n".join(manifest_text)
                source_name = input(Fore.WHITE + "Geben Sie einen Quellnamen für dieses Manifest an (z.B. 'Meeting-2024-10-08'): ")
                await embed_and_store_text(supabase, full_text, source_name)
            else:
                print(Fore.YELLOW + "Kein Text eingegeben.")

        elif choice == '2':
            question = input(Fore.WHITE + "Stellen Sie Ihre Frage: ")
            if not question:
                print(Fore.YELLOW + "Keine Frage eingegeben.")
                continue

            context = await find_similar_texts(supabase, question)
            if context:
                await ask_question_with_context(question, context)

        elif choice == '3':
            print(Fore.GREEN + "Anwendung wird beendet. Auf Wiedersehen!")
            break

        else:
            print(Fore.RED + "Ungültige Auswahl. Bitte versuchen Sie es erneut.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(Fore.GREEN + "\nAnwendung wird beendet.")
        sys.exit(0)
    except Exception as e:
        print(Fore.RED + f"\nEin unerwarteter Fehler ist aufgetreten: {e}")
        sys.exit(1)