import os
from dotenv import load_dotenv
load_dotenv()

import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-in-prod")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# session_id -> {
#   "chunks":   [{"text":..., "source":..., "chunk_id":...}],
#   "index":    faiss index,
#   "embedder": SentenceTransformer model,
#   "docs":     [{"name":..., "chunks": N}],
#   "history":  [{"role": "user"|"assistant", "content": "..."}],
# }
sessions = {}


@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    sid = session.get("sid", str(uuid.uuid4()))
    session["sid"] = sid

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()

    if ext not in (".pdf", ".txt"):
        return jsonify({"error": "Only PDF and TXT files are supported"}), 400

    prev = sessions.get(sid, {})
    existing_names = [d["name"] for d in prev.get("docs", [])]
    if filename in existing_names:
        return jsonify({"error": f"'{filename}' is already indexed in this session."}), 400

    save_path = UPLOAD_FOLDER / f"{sid}_{filename}"
    file.save(save_path)

    try:
        from rag.parser import extract_text
        from rag.chunker import chunk_text
        from rag.embedder import build_index

        text = extract_text(str(save_path))
        if not text.strip():
            return jsonify({"error": "Could not extract text from file"}), 400

        new_chunks = chunk_text(text, source=filename)

        index, model = build_index(
            new_chunks,
            existing_index=prev.get("index"),
            existing_model=prev.get("embedder"),
        )

        sessions[sid] = {
            "chunks":   prev.get("chunks", []) + new_chunks,
            "index":    index,
            "embedder": model,
            "docs":     prev.get("docs", []) + [{"name": filename, "chunks": len(new_chunks)}],
            "history":  prev.get("history", []),  # preserve history across uploads
        }

        return jsonify({
            "message": f"Ingested '{filename}' — {len(new_chunks)} chunks indexed.",
            "chunks":  len(new_chunks),
            "doc":     filename,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    sid = session.get("sid")
    if not sid or sid not in sessions:
        return jsonify({"error": "No document uploaded yet. Please upload a PDF or TXT first."}), 400

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        from rag.retriever import retrieve
        from rag.llm import answer_question

        rag_data  = sessions[sid]
        history   = rag_data.get("history", [])

        retrieved = retrieve(
            question,
            rag_data["index"],
            rag_data["chunks"],
            rag_data["embedder"],
            top_k=4,
        )

        answer = answer_question(question, retrieved, history=history)

        # Append this turn to history
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer})
        sessions[sid]["history"] = history

        # Deduplicated source list
        seen, sources = set(), []
        for c in retrieved:
            key = (c["source"], c["chunk_id"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source":   c["source"],
                    "chunk_id": c["chunk_id"],
                    "preview":  c["text"][:120] + "…",
                    "score":    round(c["score"], 3),
                })

        return jsonify({
            "answer":       answer,
            "sources":      sources,
            "history_len":  len(history) // 2,  # number of turns, for the UI badge
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    sid = session.get("sid")
    if sid and sid in sessions:
        del sessions[sid]
    return jsonify({"message": "Session cleared."})


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear only the conversation history, keep the indexed documents."""
    sid = session.get("sid")
    if sid and sid in sessions:
        sessions[sid]["history"] = []
    return jsonify({"message": "Conversation history cleared."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
