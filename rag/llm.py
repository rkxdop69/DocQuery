"""llm.py — Send retrieved context + conversation history to Gemini."""
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM = (
    "You are DocQuery, a precise document assistant. "
    "Answer the user's question using ONLY the provided context excerpts. "
    "Each excerpt is labelled with its source filename. "
    "You have access to the conversation history — use it to handle follow-up "
    "questions like 'elaborate on that' or 'what about the second point?'. "
    "If the answer isn't in the context, say so honestly. Be concise."
)

# Maximum number of past Q&A turns to include in the prompt
MAX_HISTORY_TURNS = 4


def answer_question(
    question: str,
    chunks: list[dict],
    history: list[dict],  # [{"role": "user"|"assistant", "content": "..."}]
    retries: int = 3,
) -> str:
    # Build context block
    context_parts = [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    context = "\n\n---\n\n".join(context_parts)

    # Build conversation history block (last N turns, oldest first)
    recent = history[-(MAX_HISTORY_TURNS * 2):]  # each turn = 2 entries (user + assistant)
    history_block = ""
    if recent:
        lines = []
        for msg in recent:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")
        history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    prompt = (
        f"Context:\n{context}\n\n"
        f"{history_block}"
        f"User: {question}\n\n"
        f"Answer:"
    )

    for attempt in range(retries):
        try:
            response = _client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(system_instruction=SYSTEM),
                contents=prompt,
            )
            return response.text.strip()

        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < retries - 1:
                time.sleep(15 * (attempt + 1))
                continue
            raise RuntimeError(f"Gemini error: {msg}") from e
