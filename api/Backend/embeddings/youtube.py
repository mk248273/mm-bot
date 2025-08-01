from flask import Blueprint, request, jsonify
from utils.transcriber import download_full_video, convert_to_mp3, transcribe_youtube
from utils.embeddings import save_embeddings

youtube_url = Blueprint("youtube_url", __name__)

@youtube_url.route("/youtube_url", methods=["POST"])
def process_youtube_url():
    try:
        if request.method == "POST":
            video_url = request.form.get("url")
        else:
            video_url = request.args.get("url")

        if not video_url:
            return jsonify({"error": "No YouTube URL provided"}), 400

        transcript = transcribe_youtube(video_url)

        #with open(transcript_path, "r", encoding="utf-8") as f:
           # transcript = f.read()

        save_embeddings(transcript, "youtube_index.faiss", "youtube_chunks.pkl")

        return jsonify({"message": "YouTube video processed and embedded successfully!"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
