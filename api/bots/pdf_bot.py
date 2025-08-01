from flask import Blueprint, request, jsonify
from utils.embeddings import load_pdf_embeddings, answer_from_pdf
import os

pdf_bot = Blueprint("pdf_bot", __name__)

@pdf_bot.route("/pdf_bot", methods=["POST"])
def ask_from_pdf():
    try:
        question = request.form.get("question")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Use consistent vector paths from the upload route
        pdf_index_path = "vector/pdf.index"
        chunks_path = "vector/pdf_chunks.pkl"

        # Ensure the index and chunks exist
        if not os.path.exists(pdf_index_path) or not os.path.exists(chunks_path):
            return jsonify({"error": "PDF embedding not found. Please upload the PDF first."}), 404

        # Load index and chunks
        index, chunks = load_pdf_embeddings(pdf_index_path, chunks_path)

        # Answer the question
        answer = answer_from_pdf(question, pdf_index_path, chunks_path)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
