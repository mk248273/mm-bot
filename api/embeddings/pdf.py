from flask import Blueprint, request, jsonify
from utils.shared_pdf_utils import extract_pdf_text
from utils.embeddings import save_pdf_embeddings
import os

pdf_embed = Blueprint("pdf_embed", __name__)

@pdf_embed.route("/pdf_upload", methods=["POST"])
def upload_pdf():
    try:
        uploaded_file = request.files.get("file")  # Match Postman key
        if not uploaded_file:
            return jsonify({"error": "No PDF file provided"}), 400

        # Extract PDF text
        text = extract_pdf_text(uploaded_file)

        # Ensure 'vector' directory exists
        os.makedirs("vector", exist_ok=True)

        # Save embeddings
        msg = save_pdf_embeddings(text, "vector/pdf.index", "vector/pdf_chunks.pkl")

        return jsonify({
            "message": "PDF uploaded and processed successfully.",
            "embedding_status": msg
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
