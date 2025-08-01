from flask import Blueprint, request, jsonify
from utils.embeddings import answer_question
from groq import Groq
import os

text_bot = Blueprint("text_bot", __name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@text_bot.route("/text_bot", methods=["POST"])
def ask_from_text():
    try:
        question = request.form.get("question")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        #  Use full correct paths
        index_path = r"C:\Users\FIREFLY LAPTOPS\Desktop\QNA\text_index.faiss"
        chunk_path = r"C:\Users\FIREFLY LAPTOPS\Desktop\QNA\text_chunks.pkl"

        #  Check if files exist
        if not os.path.exists(index_path) or not os.path.exists(chunk_path):
            return jsonify({"error": "Text embeddings not found. Please upload text first."}), 404

        # Run the embedding-based answer function
        context = answer_question(index_path, chunk_path, question)

        #  Prepare LLM prompt
        prompt = f"You are given the following passage:\n{context}\n\nBased on it, answer this question: {question}\nAnswer:"

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You're a helpful assistant for analyzing uploaded text."},
                {"role": "user", "content": prompt}
            ]
        )

        return jsonify({"answer": response.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
