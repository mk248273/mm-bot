from flask import Blueprint, request, jsonify
from utils.text import read_text_file, read_text_data
from utils.embeddings import save_embeddings

# Create the Blueprint
text_upload = Blueprint("text_upload", __name__)

# Define the route for uploading text
@text_upload.route("/text_upload", methods=["POST"])
def handle_text_upload():
    try:
        # Get either uploaded file or pasted text
        uploaded_file = request.files.get("file")
        text_data = request.form.get("text")

        if uploaded_file:
            text = read_text_file(uploaded_file)
        elif text_data:
            text = read_text_data(text_data)
        else:
            return jsonify({"error": "No input provided"}), 400

        # Save embeddings
        message = save_embeddings(text, "text_index.faiss", "text_chunks.pkl")
        return jsonify({"message": message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
