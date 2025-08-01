import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Import blueprints
from bots.pdf_bot import pdf_bot
from bots.website_bot import website_bot
from bots.text_bot import text_bot
from bots.youtube_bot import youtube_bot

# Import embedding blueprints
from embeddings.text import text_upload
from embeddings.pdf import pdf_embed
from embeddings.youtube import youtube_url
from embeddings.website import website_url

# App initialization
app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(pdf_bot)
app.register_blueprint(website_bot)
app.register_blueprint(text_bot)
app.register_blueprint(youtube_bot)

# Register embedding blueprints
app.register_blueprint(text_upload)
app.register_blueprint(pdf_embed)
app.register_blueprint(youtube_url)
app.register_blueprint(website_url)

# Frontend routes
@app.route("/", methods=["GET"])
def home():
    """Serve the main frontend interface"""
    return render_template('index.html')

@app.route("/api/status", methods=["GET"])
def api_status():
    """API status endpoint for frontend to check backend connectivity"""
    return {"message": "Backend is running!", "status": "online"}

# Static files route (if you need to serve additional CSS/JS files)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return {"error": "Internal server error"}, 500

# Run server
if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')