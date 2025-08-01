from flask import Blueprint, request, jsonify
from utils.embeddings import load_youtube_embeddings, answer_from_youtube, save_youtube_embeddings
from utils.transcriber import transcribe_youtube
from groq import Groq
import os

youtube_bot = Blueprint("youtube_bot", __name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ensure_parent_dir(path, base_dir="embeddings"):
    """Ensure that path is safely created inside a controlled base directory."""
    dir_name = os.path.dirname(path)
    if dir_name and base_dir in os.path.abspath(dir_name):
        os.makedirs(dir_name, exist_ok=True)

@youtube_bot.route("/youtube_bot", methods=["POST"])
def ask_from_youtube():
    try:
        question = request.form.get("question")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Paths
        index_path = "youtube_index.faiss"
        chunk_path = "youtube_chunks.pkl"

        # Ensure directory (safe)
        ensure_parent_dir(chunk_path)

        # Get answer
        answer = answer_from_youtube(question, index_path, chunk_path)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
