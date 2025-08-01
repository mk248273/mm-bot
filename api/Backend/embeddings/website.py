from flask import Blueprint, request, jsonify
from utils.website import extract_website_text
from utils.embeddings import save_embeddings

website_url = Blueprint("website_url", __name__)

@website_url.route("/website_url", methods=["POST"])
def process_website_url():
    url = request.form.get("url")
    try:
        text = extract_website_text(url)
        message = save_embeddings(text, "website_index.faiss", "website_chunks.pkl")
        return jsonify({"message": text})
    except Exception as e:
        return jsonify({"error": str(e)})
