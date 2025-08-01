from flask import Flask
from flask_cors import CORS
from bots.pdf_bot import pdf_bot
from embeddings.pdf import pdf_embed

app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(pdf_bot)
app.register_blueprint(pdf_embed)

@app.route("/", methods=["GET"])
def home():
    return {"message": "Q&A Backend Running"}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
