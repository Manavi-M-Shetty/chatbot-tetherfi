# src/index_docs.py
import os
import json
import base64

from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import requests

# Local embedding model for text and captions
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def chunk_text(text, chunk_size=1500, overlap=200):
    """Simple character-based chunking."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_texts(texts):
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings.tolist()


def read_pdf_pages(path):
    """Return list of dicts: [{'page': 1, 'text': '...'}, ...]"""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages


def create_chunks_from_pages(pages, doc_id, doc_name):
    """Chunk text per page so we always know which page each chunk came from."""
    chunks = []
    global_idx = 0
    for page in pages:
        page_num = page["page"]
        page_text = page["text"]
        page_chunks = chunk_text(page_text, chunk_size=1500, overlap=200)
        for local_idx, chunk in enumerate(page_chunks):
            chunks.append({
                "id": f"{doc_id}_p{page_num}_c{local_idx}",
                "text": chunk,
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": page_num,
                    "chunk_index": global_idx,
                },
            })
            global_idx += 1
    return chunks


def extract_images_from_pdf(pdf_path, doc_id):
    """
    Extract images from each page and save:
      - image files under data/images/{doc_id}/
      - mapping page -> [image_paths] to data/images/{doc_id}_images.json
    """
    images_root = os.path.join("data", "images")
    os.makedirs(images_root, exist_ok=True)

    doc_image_dir = os.path.join(images_root, doc_id)
    os.makedirs(doc_image_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page_images_map: dict[int, list[str]] = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        image_list = page.get_images(full=True)
        img_paths = []

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            img_filename = f"{doc_id}_p{page_num}_img{img_index}.{image_ext}"
            img_path = os.path.join(doc_image_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            img_paths.append(img_path)

        if img_paths:
            page_images_map[page_num] = img_paths

    doc.close()

    # Save JSON mapping: { "1": ["path1", "path2"], "2": [...], ... }
    mapping_path = os.path.join(images_root, f"{doc_id}_images.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(page_images_map, f, indent=2)

    print(f"Extracted images for {doc_id} to {doc_image_dir}")
    return page_images_map


def caption_image_with_llava(image_path: str) -> str:
    """
    Generate a caption for an image using local LLaVA via Ollama.
    Make sure you have run: `ollama pull llava`.
    """
    url = "http://localhost:11434/api/generate"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are helping to describe images from a technical product documentation. "
       "Give a **very short** 1–2 sentence caption for this image from technical product documentation. "
        "Mention what kind of thing it is (e.g., UI screenshot, architecture diagram, flow chart) "
        "and the main elements visible."
    )

    data = {
        "model": "llava:7b",   # change if you pulled another variant
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    resp = requests.post(url, json=data, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    caption = result.get("response", "").strip()
    return caption or "Image from documentation."


def index_image_captions(doc_id: str, doc_name: str, page_images_map: dict[int, list[str]], chroma_client):
    """
    For each image, generate a caption and index it in a separate Chroma collection.
    """
    image_collection = chroma_client.get_or_create_collection("product_images")

    image_ids = []
    captions = []
    metadatas = []

    for page, img_paths in page_images_map.items():
        for i, img_path in enumerate(img_paths):
            print(f"Captioning image on page {page}: {img_path}")
            try:
                caption = caption_image_with_llava(img_path)
            except Exception as e:
                print(f"  [WARNING] Failed to caption {img_path}: {e}")
                caption = "Image from documentation."

            img_id = f"{doc_id}_p{page}_img{i+1}"
            image_ids.append(img_id)
            captions.append(caption)
            metadatas.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": page,
                "image_path": img_path,
            })

    if not captions:
        print("No images to index.")
        return

    print("Embedding image captions...")
    caption_embeddings = embed_texts(captions)

    image_collection.add(
        ids=image_ids,
        documents=captions,
        metadatas=metadatas,
        embeddings=caption_embeddings,
    )

    print(f"Indexed {len(captions)} image captions.")


def main():
    pdf_path = os.path.join("data", "raw", "product1.pdf")
    doc_id = "product1"
    doc_name = "Product 1"  

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    # --- 1. Read text pages and create chunks with page metadata ---
    pages = read_pdf_pages(pdf_path)
    chunks = create_chunks_from_pages(pages, doc_id, doc_name)
    print(f"Total text chunks: {len(chunks)}")

    texts = [c["text"] for c in chunks]
    text_embeddings = embed_texts(texts)

    # --- 2. Store text chunks+embeddings in Chroma ---
    chroma_client = chromadb.PersistentClient(path="db")
    text_collection = chroma_client.get_or_create_collection(name="product_docs")

    text_collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
        embeddings=text_embeddings,
    )

    print("Text indexing done.")

    # --- 3. Extract images and index their captions ---
    page_images_map = extract_images_from_pdf(pdf_path, doc_id)
    if page_images_map:
        index_image_captions(doc_id, doc_name, page_images_map, chroma_client)
    else:
        print("No images found in PDF.")

    print("All indexing done.")


if __name__ == "__main__":
    main()