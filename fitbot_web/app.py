import os
import json
import torch
from flask import Flask, render_template, request, session, redirect, url_for

# Flask setup 
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "your_secret_key_here"  # replace this in production

# Paths
HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "model")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
EMB_PATH = os.path.join(MODEL_DIR, "embedding_matrix.pt")
CKPT_PATH = os.path.join(MODEL_DIR, "best_gru_model.pth")

# Load vocabulary & embeddings
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    word2index = json.load(f)
index2word = {int(v): k for k, v in word2index.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_matrix = torch.load(EMB_PATH, map_location=device)

# Import model class & helpers
from model.fitbot import FitnessGRUBot, encode_text, generate_response

# Instantiate and load model
bot = FitnessGRUBot(
    vocab_size=len(word2index),
    embed_dim=100,
    hidden_dim=128,
    num_layers=2,
    dropout=0.2,
    pretrained_embeddings=embedding_matrix
).to(device)

print("Loading model checkpoint...")
bot.load_state_dict(torch.load(CKPT_PATH, map_location=device))
bot.eval()
print("✅ Model loaded.")

# Routes 
@app.route("/")
def home():
    session.pop("messages", None)
    return render_template("home.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()

        if user_input:
            bot_reply = generate_response(user_input)
            session["messages"].append((user_input, bot_reply))
            session.modified = True

    return render_template("chat.html", messages=session["messages"])


@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("messages", None)
    return redirect(url_for("chat"))


# Run server 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
