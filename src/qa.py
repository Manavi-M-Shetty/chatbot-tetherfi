import os
import json

import chromadb
from sentence_transformers import SentenceTransformer
import requests
from PIL import Image  # for logo-like detection based on size


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ----- PATHS (relative to project root) -----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, "db")
IMAGES_ROOT = os.path.join(PROJECT_ROOT, "data", "images")
# --------------------------------------------

# Connect to Chroma using absolute DB path
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection("product_docs")


def embed_texts(texts):
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()


def build_context(documents, metadatas):
    parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        title = meta.get("doc_name", "Unknown Document")
        page = meta.get("page", "unknown")
        chunk_index = meta.get("chunk_index", i)
        header = f"[{title} – page {page}, chunk {chunk_index}]"
        parts.append(f"{header}\n{doc}")
    return "\n\n---\n\n".join(parts)


def load_page_images(doc_id: str) -> dict[int, list[str]]:
    """
    Load JSON mapping {page_number: [image_path, ...]} for this document.
    Image paths are stored relative to project root.
    """
    mapping_path = os.path.join(IMAGES_ROOT, f"{doc_id}_images.json")
    if not os.path.exists(mapping_path):
        return {}

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON keys are strings; convert to int
    return {int(k): v for k, v in data.items()}


def get_abs_image_path(path: str) -> str:
    """Resolve image path relative to project root if needed."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def is_logo_like(path: str, page: int | None = None) -> bool:
    """
    Heuristic to detect logo-like images:
    - very small images
    - or small & nearly square
    - optionally more aggressive on first pages
    """
    abs_path = get_abs_image_path(path)
    try:
        with Image.open(abs_path) as img:
            w, h = img.size
    except Exception:
        # If we can't open it, don't treat it as logo
        return False

    area = w * h
    aspect = w / h if h else 0.0

    # 1) Very small images → likely logo/icon
    if area < 150_000:
        return True

    # 2) Small & almost square (logos often are)
    if area < 300_000 and 0.8 <= aspect <= 1.25:
        return True

    # 3) Slightly more aggressive on first pages
    if page is not None and page <= 2 and area < 400_000:
        return True

    return False


def is_greeting(text: str) -> bool:
    """Simple greeting detector to answer 'hi', 'hello', etc. nicely."""
    t = text.strip().lower()
    if not t:
        return False
    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    ]
    return any(t.startswith(g) for g in greetings)


def format_chat_history(chat_history, max_messages: int = 10) -> str:
    """
    Convert chat history into a compact text form for the LLM.
    Keeps only the last `max_messages` messages (user + assistant).
    """
    if not chat_history:
        return ""

    recent = chat_history[-max_messages:]
    lines = []
    for msg in recent:
        role = msg.get("role", "")
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            continue

        content = msg.get("content", "").strip()
        if not content:
            continue

        lines.append(f"{prefix}: {content}")

    return "\n".join(lines)


def condense_question(question: str, chat_history=None) -> str:
    """
    For follow-up questions like 'What about its architecture?',
    rewrite them into standalone questions using the chat history.
    If history is empty or the rewrite fails, return the original question.
    """
    if not chat_history:
        return question

    history_text = format_chat_history(chat_history, max_messages=10)
    if not history_text.strip():
        return question

    url = "http://localhost:11434/api/chat"
    data = {
        "model": "llama3",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You reformulate follow-up questions into standalone questions.\n"
                    "Given a chat history and a follow-up question, rewrite the question so it can\n"
                    "be understood without the history. Keep all specific details and be concise.\n"
                    "Output ONLY the rewritten question, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Chat history:\n{history_text}\n\n"
                    f"Follow-up question: {question}\n\n"
                    "Rewritten standalone question:"
                ),
            },
        ],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        rewritten = result["message"]["content"].strip()
        return rewritten or question
    except Exception as e:
        print(f"[WARNING] condense_question failed: {e}")
        return question


def call_local_llm(
    question: str,
    context: str,
    chat_history=None,
    standalone_question: str | None = None,
) -> str:
    """
    Doc-based answering:
    - MUST use only the provided context.
    - May use chat history ONLY to resolve references / keep continuity,
      not as a new factual source.
    - If the context is insufficient or the question is unrelated,
      it must respond EXACTLY with: DOCS_UNKNOWN
      (no extra text).
    """
    history_text = format_chat_history(chat_history) if chat_history else ""

    # Build user message
    parts = []
    if history_text:
        parts.append(f"Conversation so far:\n{history_text}")

    parts.append(f"User's current question: {question}")

    if standalone_question and standalone_question != question:
        parts.append(
            "A standalone reformulation of the question for your reference:\n"
            f"{standalone_question}"
        )

    parts.append(
        "Context from documentation (the ONLY factual source you may use):\n"
        f"{context}\n\n"
        "Answer the user's current question using only the context above.\n"
        "Use the conversation history only to resolve pronouns and keep a natural tone.\n"
        "If you cannot answer from this context, respond with exactly: DOCS_UNKNOWN"
    )

    user_content = "\n\n".join(parts)

    url = "http://localhost:11434/api/chat"
    data = {
        "model": "llama3",  # or "llama3.1:8b", etc.
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a chatbot for internal company product documentation.\n"
                    "- You MUST answer using ONLY the provided documentation context.\n"
                    "- Conversation history is ONLY for understanding references and tone;\n"
                    "  do NOT treat it as a new factual source.\n"
                    "- If the context does not contain enough information to answer the question,\n"
                    "  or if the question is unrelated to the documentation, respond with EXACTLY\n"
                    "  the single token: DOCS_UNKNOWN\n"
                    "- Do NOT add any other words when you respond with DOCS_UNKNOWN."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "stream": False,
    }

    resp = requests.post(url, json=data)
    resp.raise_for_status()
    result = resp.json()
    return result["message"]["content"].strip()


def answer_from_model_only(question: str, chat_history=None) -> str:
    """
    Fallback: answer from the model's own general knowledge,
    without using internal documentation. Works fully offline.
    Uses chat history for continuity.
    """
    history_text = format_chat_history(chat_history) if chat_history else ""

    parts = []
    if history_text:
        parts.append(f"Conversation so far:\n{history_text}")
    parts.append(f"User's current question: {question}")

    user_content = "\n\n".join(parts)

    url = "http://localhost:11434/api/chat"
    data = {
        "model": "llama3",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You may answer using your own general knowledge.\n"
                    "If the question seems to be about this company's internal products or documents,\n"
                    "and you do not have that information, say that you don't have access to the\n"
                    "internal documentation, but you can provide a general answer if appropriate."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=data)
        resp.raise_for_status()
        result = resp.json()
        return result["message"]["content"].strip()
    except Exception as e:
        print(f"[WARNING] Model-only LLM call failed: {e}")
        return "I couldn't find relevant information in the documentation."


def is_docs_unknown_response(answer_text: str) -> bool:
    """
    Detect when the doc-based LLM is signalling that the docs are insufficient.
    Many models ignore the 'respond exactly DOCS_UNKNOWN' instruction and instead
    wrap it in a longer sentence, e.g.:
      "I cannot answer from the docs, so my response is: DOCS_UNKNOWN"

    This treats ANY occurrence of 'DOCS_UNKNOWN' (case-insensitive) as a signal
    to use the fallback model.
    """
    if not answer_text:
        return False
    t = answer_text.strip().upper()
    if t == "DOCS_UNKNOWN":
        return True
    if "DOCS_UNKNOWN" in t:
        return True
    return False


def answer_question(question: str, doc_id: str | None = None, chat_history=None):
    # --- 0. Handle simple greetings without RAG ---
    if is_greeting(question):
        return (
            "Hello! I’m a documentation assistant. "
            "Ask me a question about the product, for example:\n\n"
            "- What are the main capabilities?\n"
            "- How does the architecture work?\n"
            "- Does it support integration with X?\n",
            [],
        )

    # --- 1. Optionally rewrite follow-up question into standalone form ---
    standalone_question = condense_question(question, chat_history)

    # --- 2. Embed question (standalone form) locally ---
    q_embedding = embed_texts([standalone_question])[0]

    # --- 3. Retrieve relevant chunks from documentation ---
    where_filter = {"doc_id": doc_id} if doc_id else None

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=5,
        where=where_filter,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # If we retrieved nothing at all from docs -> fallback to model knowledge
    if not documents:
        model_answer = answer_from_model_only(question, chat_history=chat_history)
        return model_answer, []

    # --- 4. Build context and ask doc-only LLM (with history) ---
    context = build_context(documents, metadatas)
    answer_text = call_local_llm(
        question=question,
        context=context,
        chat_history=chat_history,
        standalone_question=standalone_question,
    )

    # If LLM says it cannot answer from docs -> fallback to model knowledge
    if is_docs_unknown_response(answer_text):
        model_answer = answer_from_model_only(question, chat_history=chat_history)
        return model_answer, []

    # --- 5. Collect relevant images based on pages, auto-skip logos ---
    image_paths: list[str] = []
    if doc_id:
        page_images_map = load_page_images(doc_id)

        # pages where relevant chunks came from
        pages = {meta.get("page") for meta in metadatas if "page" in meta}

        for p in pages:
            if p not in page_images_map:
                continue

            for img_path in page_images_map[p]:
                if is_logo_like(img_path, page=p):
                    continue
                image_paths.append(img_path)

        # remove duplicates, keep sorted
        image_paths = sorted(set(image_paths))

    return answer_text, image_paths


if __name__ == "__main__":
    print("Ask questions about product1. Type 'exit' to quit.")
    chat_history = []
    while True:
        q = input("Q: ")
        if q.lower() in ("exit", "quit"):
            break

        ans, imgs = answer_question(q, doc_id="product1", chat_history=chat_history)
        print(f"A: {ans}\n")

        if imgs:
            print("Relevant images (file paths):")
            for img in imgs:
                print(" -", get_abs_image_path(img))
        else:
            print("No relevant images found.")

        # Update CLI chat history
        chat_history.append({"role": "user", "content": q, "images": []})
        chat_history.append({"role": "assistant", "content": ans, "images": imgs})

        print()