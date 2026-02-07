from flask import Flask, request, jsonify, send_file
import json
import os
from datetime import datetime, timezone

app = Flask(__name__)

DB_FILE = "./jsondb.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/save", methods=["POST"])
def save():
    entry = request.json
    # استخدام UTC aware لتجنب التحذير
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()

    db = load_db()
    db.append(entry)
    save_db(db)

    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
