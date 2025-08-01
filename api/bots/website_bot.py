from flask import Blueprint, request, jsonify
from utils.embeddings import load_website_embeddings, answer_from_website

website_bot = Blueprint("website_bot", __name__)

@website_bot.route("/website_bot", methods=["POST"])
def ask_from_website():
    try:
        question = request.form.get("question")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        
        chunks_path = r"website_chunks.pkl"
        website_index_path = r"website_index.faiss"

        # Load embeddings and chunks
        index, chunks = load_website_embeddings(website_index_path, chunks_path)

        # Get answer from website data
        answer = answer_from_website(question, website_index_path, chunks_path)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
