import os
import json
import uuid
from datetime import datetime

import streamlit as st

from qa import answer_question  # uses your RAG + images + multi-turn


# ---------- PATHS ----------
# Resolve project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_LOG_DIR = os.path.join(PROJECT_ROOT, "chat_logs")
os.makedirs(CHAT_LOG_DIR, exist_ok=True)
# ---------------------------


st.set_page_config(page_title="Product Documentation Chatbot", layout="wide")

st.title("Product Documentation Chatbot")

st.markdown(
    "Ask questions about the product documentation. "
    "Answers are generated locally using your docs + a local LLM (Ollama)."
)

# For now you only have one doc; later you can add more IDs here
PRODUCT_IDS = {
    "Product 1": "product1",
    # "Product 2": "product2",
}

# Use a key so we can control this from loaded chats
product_label = st.selectbox(
    "Select product document",
    list(PRODUCT_IDS.keys()),
    key="product_select",
)
product_id = PRODUCT_IDS[product_label]

# ---- Session state: per-session ID and chat history ----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # each item: {"role": "user"/"assistant", "content": str, "images": [paths]}
    st.session_state.messages = []


def resolve_image_path(img_path: str) -> str:
    """Resolve image path relative to project root if needed."""
    if os.path.isabs(img_path):
        return img_path
    return os.path.join(PROJECT_ROOT, img_path)


def save_chat_history(product_id: str):
    """
    Persist the current session's chat history to a JSON file.
    One file per (session, product), updated on every turn.
    """
    session_id = st.session_state.session_id
    log_path = os.path.join(CHAT_LOG_DIR, f"{product_id}_{session_id}.json")
    data = {
        "session_id": session_id,
        "product_id": product_id,
        "messages": st.session_state.messages,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reset_chat():
    """Start a completely new conversation."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def list_chat_logs(filter_product_id: str | None = None):
    """
    Return a list of existing chat logs for a given product, newest first.

    Each item is:
        {"label": str, "path": str}
    """
    logs = []
    if not os.path.exists(CHAT_LOG_DIR):
        return logs

    for filename in os.listdir(CHAT_LOG_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(CHAT_LOG_DIR, filename)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        pid = data.get("product_id")
        if filter_product_id and pid != filter_product_id:
            continue

        messages = data.get("messages", [])

        # Use first user message as a short title
        title = "(empty conversation)"
        for m in messages:
            if m.get("role") == "user" and m.get("content"):
                title = m["content"].strip()
                break
        if len(title) > 60:
            title = title[:57] + "..."

        mtime = os.path.getmtime(path)
        dt_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        label = f"{dt_str} – {title}"
        logs.append({"label": label, "path": path})

    # Newest first
    logs.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
    return logs


def load_selected_chat():
    """
    Load the chat selected in the sidebar into session_state.messages.
    """
    product_label = st.session_state.get("product_select")
    if not product_label:
        return

    product_id = PRODUCT_IDS[product_label]
    logs = list_chat_logs(filter_product_id=product_id)
    selected_label = st.session_state.get("prev_chat_select")
    if not selected_label:
        return

    log = next((l for l in logs if l["label"] == selected_label), None)
    if not log:
        return

    path = log["path"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Failed to load chat: {e}")
        return

    # Restore session ID and messages
    st.session_state.session_id = data.get("session_id", str(uuid.uuid4()))
    st.session_state.messages = data.get("messages", [])

    # Ensure the product selector matches the conversation's product
    loaded_pid = data.get("product_id")
    if loaded_pid:
        for label, pid in PRODUCT_IDS.items():
            if pid == loaded_pid:
                st.session_state.product_select = label
                break


# ---------- SIDEBAR: conversation controls ----------
with st.sidebar:
    st.header("Conversations")

    st.button("Start new chat", on_click=reset_chat)

    logs = list_chat_logs(filter_product_id=product_id)
    if logs:
        labels = [log["label"] for log in logs]
        st.selectbox(
            "Previous chats",
            labels,
            key="prev_chat_select",
        )
        st.button("Load selected chat", on_click=load_selected_chat)
    else:
        st.caption("No previous chats for this product yet.")
# ----------------------------------------------------


# -------- Render existing chat history --------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            for img_path in msg.get("images", []):
                abs_path = resolve_image_path(img_path)
                if os.path.exists(abs_path):
                    st.image(abs_path, width=500)
                else:
                    st.caption(f"(Image file not found: {abs_path})")
# ----------------------------------------------


# -------- Chat input (new question) --------
if prompt := st.chat_input("Ask a question about the selected product…"):
    # Add user message to history
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "images": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from your RAG pipeline WITH chat history (multi-turn)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass all previous messages except the just-added user message
            answer_text, image_paths = answer_question(
                prompt,
                doc_id=product_id,
                chat_history=st.session_state.messages[:-1],
            )
        st.markdown(answer_text)

        for img_path in image_paths:
            abs_path = resolve_image_path(img_path)
            if os.path.exists(abs_path):
                st.image(abs_path, width=500)

    # Store assistant message + images in history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "images": image_paths}
    )

    # Persist chat history for this session + product
    save_chat_history(product_id)
# -------------------------------------------