import asyncio
import logging
import os
import sys
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
from typing import Optional

import asyncpg
import google.generativeai as genai
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector
import json

# --- Initialization ---
# Load environment variables
load_dotenv(".env")
load_dotenv(".env.local", override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProtokollFuererApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Protokollführer")
        self.conn: Optional[asyncpg.Connection] = None

        # --- Environment Variable Setup ---
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.DB_CONNECTION_URI = os.getenv("DB_CONNECTION_URI")

    async def initialize_app(self):
        """Initializes the application, checks connections, and sets up the schema."""
        self.log_message("Initialisiere Anwendung...")
        if not self.check_env_variables():
            return False
        if not await self.get_db_connection():
            return False
        if not await self.ensure_schema():
            return False

        self.log_message("Anwendung erfolgreich initialisiert.", level="SUCCESS")
        return True

    def check_env_variables(self):
        """Checks for necessary environment variables."""
        if not self.GEMINI_API_KEY or not self.DB_CONNECTION_URI:
            messagebox.showerror("Fehlende Konfiguration", "Kritische Umgebungsvariablen (DB_CONNECTION_URI oder GEMINI_API_KEY) fehlen.")
            self.log_message("Kritische Umgebungsvariablen fehlen.", level="ERROR")
            return False
        genai.configure(api_key=self.GEMINI_API_KEY)
        self.log_message("Umgebungsvariablen erfolgreich geladen.")
        return True

    async def get_db_connection(self):
        """Establishes a database connection."""
        try:
            self.conn = await asyncpg.connect(self.DB_CONNECTION_URI)
            await register_vector(self.conn)
            self.log_message("Datenbankverbindung erfolgreich hergestellt.")
            return True
        except Exception as e:
            messagebox.showerror("Datenbankfehler", f"Verbindung zur Datenbank fehlgeschlagen: {e}")
            self.log_message(f"Verbindung zur Datenbank fehlgeschlagen: {e}", level="ERROR")
            return False

    async def ensure_schema(self):
        """Ensures the database schema and pgvector extension are set up."""
        try:
            self.log_message("Überprüfe Datenbankschema...")
            await self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await self.conn.execute("""
                CREATE TABLE IF NOT EXISTS protokolle (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding VECTOR(768),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            self.log_message("Datenbankschema verifiziert und bereit.")
            return True
        except Exception as e:
            messagebox.showerror("Schema-Fehler", f"Fehler beim Sicherstellen des Datenbankschemas: {e}")
            self.log_message(f"Fehler beim Sicherstellen des Datenbankschemas: {e}", level="ERROR")
            return False

    def log_message(self, message: str, level: str = "INFO"):
        """Logs a message to the GUI and the console."""
        logging.info(message)
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, f"[{level}] {message}\n")
        self.output_text.config(state='disabled')
        self.output_text.see(tk.END)

    async def handle_save_manifest(self):
        """Handles the 'SPEICHERN' button click."""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Leere Eingabe", "Das Eingabefeld ist leer.")
            return

        source_name = simpledialog.askstring("Quellenname", "Geben Sie einen Quellnamen für dieses Manifest an:", parent=self.root)
        if not source_name:
            source_name = "gui_input"

        self.log_message(f"Speichere Manifest von Quelle: {source_name}")
        await self.embed_and_store_text(text, source_name)

    async def handle_rag_query(self):
        """Handles the 'FRAGE STELLEN' button click."""
        question = self.input_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Leere Frage", "Das Eingabefeld ist leer.")
            return

        self.log_message(f"Anfrage wird bearbeitet: {question}")
        await self.perform_rag_query(question)

    async def embed_and_store_text(self, text: str, source: str) -> bool:
        """Generates embedding for text and stores it in the database."""
        self.log_message(f"Generiere Embedding für Quelle: {source}...")
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
            metadata_json = json.dumps(metadata)

            self.log_message("Speichere Text und Embedding in der Datenbank...")
            await self.conn.execute(
                "INSERT INTO protokolle (text, embedding, metadata) VALUES ($1, $2, $3)",
                text,
                embedding,
                metadata_json
            )
            self.log_message(f"Manifest von Quelle '{source}' erfolgreich gespeichert.")
            return True
        except Exception as e:
            self.log_message(f"Fehler beim Einbetten oder Speichern: {e}", level="ERROR")
            messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten: {e}")
            return False

    async def perform_rag_query(self, question: str):
        """Handles the entire RAG cycle from user input to final answer."""
        try:
            self.log_message("Generiere Vektor-Embedding für die Frage...")
            embedding_model = "text-embedding-004"
            embedding_response = await asyncio.to_thread(
                genai.embed_content,
                model=embedding_model,
                content=question,
                task_type="RETRIEVAL_QUERY"
            )
            query_embedding = embedding_response['embedding']

            self.log_message("Suche nach relevanten Dokumenten in der Datenbank...")
            retrieved_records = await self.conn.fetch(
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
                self.log_message("Keine relevanten Dokumente im Vektor-Index gefunden.", level="WARNING")
                self.log_message("\n=== ANTWORT ===\nIch konnte keine relevanten Informationen zu Ihrer Frage finden.\n===============\n")
                return

            self.log_message(f"{len(retrieved_records)} relevante Dokumente gefunden. Erstelle Kontext...")
            context_parts = []
            for rec in retrieved_records:
                source = rec['source'] or 'Unbekannt'
                similarity = rec['similarity']
                text_content = rec['text']
                context_part = (
                    f"Quelle: {source}, Ähnlichkeit: {similarity:.4f}\n"
                    f"---\n"
                    f"{text_content}"
                )
                context_parts.append(context_part)
            context = "\n\n".join(context_parts)

            self.log_message("Generiere finale Antwort mit dem Gemini-Modell...")
            model = genai.GenerativeModel('gemini-2.5-flash')
            system_prompt = (
                "Du bist ein spezialisierter Assistent für Protokoll-Analyse. "
                "Deine Aufgabe ist es, die folgende Frage prägnant und ausschließlich basierend auf dem bereitgestellten Kontext zu beantworten. "
                "Fasse die relevanten Informationen zusammen. Gib ehrlich zu, wenn die Antwort nicht im Kontext enthalten ist. "
                "Verwende keine externen Wissensquellen."
            )
            final_prompt = f"{system_prompt}\n\nKONTEXT:\n{context}\n\nFRAGE:\n{question}"

            response = await asyncio.to_thread(model.generate_content, final_prompt)

            self.log_message("\n=== ANTWORT ===")
            self.log_message(response.text)
            self.log_message("===============\n")

        except Exception as e:
            self.log_message(f"Ein Fehler ist während der Abfrageverarbeitung aufgetreten: {e}", level="ERROR")
            messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten: {e}")

    def create_widgets(self):
        """Creates and places the GUI widgets."""
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input field
        input_label = tk.Label(main_frame, text="Eingabe (Manifest oder Frage):")
        input_label.pack(anchor="w")
        self.input_text = scrolledtext.ScrolledText(main_frame, height=15, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))

        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        save_button = tk.Button(button_frame, text="SPEICHERN", command=lambda: asyncio.create_task(self.handle_save_manifest()))
        save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        query_button = tk.Button(button_frame, text="FRAGE STELLEN", command=lambda: asyncio.create_task(self.handle_rag_query()))
        query_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # Output field
        output_label = tk.Label(main_frame, text="Logs / Antwort:")
        output_label.pack(anchor="w")
        self.output_text = scrolledtext.ScrolledText(main_frame, height=15, wrap=tk.WORD, state='disabled')
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    async def run_tk_async(self):
        """Runs the Tkinter event loop within the asyncio loop."""
        try:
            while True:
                self.root.update()
                await asyncio.sleep(0.05)  # Update 20 times per second
        except tk.TclError as e:
            if "application has been destroyed" not in str(e):
                raise

# --- Main Execution ---
async def main():
    root = tk.Tk()
    app = ProtokollFuererApp(root)
    app.create_widgets()

    # Initialize the app
    if not await app.initialize_app():
        root.destroy()
        return

    # Start the async event loop for Tkinter
    await app.run_tk_async()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application terminated by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"An unexpected error occurred: {e}")
        sys.exit(1)