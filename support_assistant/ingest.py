"""
ingest.py [*] Module 3: Support Assistant
Loads all 8 Zepto policy documents, chunks them, embeds with all-MiniLM-L6-v2,
and stores embeddings in ChromaDB. Run this once before starting the FastAPI server.
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Document metadata [*] maps filename to a human-readable policy name
DOC_METADATA = {
    "doc_01.txt": "Delivery Policy",
    "doc_02.txt": "Returns & Refunds",
    "doc_03.txt": "Membership Tiers",
    "doc_04.txt": "Order Tracking",
    "doc_05.txt": "Order Cancellation Policy",
    "doc_06.txt": "Damaged or Missing Items",
    "doc_07.txt": "Gift Cards",
    "doc_08.txt": "Customer Support Hours",
}


def load_documents(docs_dir: str) -> list[dict]:
    """Load all .txt documents from the docs directory."""
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            doc_id = filename.replace(".txt", "")
            documents.append({
                "id": doc_id,
                "filename": filename,
                "policy_name": DOC_METADATA.get(filename, filename),
                "content": content,
            })
    print(f"  Loaded {len(documents)} documents from {docs_dir}")
    return documents


def chunk_documents(documents: list[dict], chunk_size: int = 500) -> list[dict]:
    """
    Chunk documents. Given the small size of each policy doc (~200-400 chars),
    we use per-document chunking (each doc = one chunk). For longer docs we
    split at chunk_size character boundaries, preserving whole sentences where possible.
    """
    chunks = []
    for doc in documents:
        content = doc["content"]
        if len(content) <= chunk_size:
            # Single chunk = entire document
            chunks.append({
                "id": doc["id"] + "_chunk_0",
                "doc_id": doc["id"],
                "policy_name": doc["policy_name"],
                "text": content,
                "chunk_index": 0,
            })
        else:
            # Split into multiple chunks
            words = content.split()
            current_chunk = []
            current_len = 0
            chunk_index = 0
            for word in words:
                current_chunk.append(word)
                current_len += len(word) + 1
                if current_len >= chunk_size:
                    chunks.append({
                        "id": f"{doc['id']}_chunk_{chunk_index}",
                        "doc_id": doc["id"],
                        "policy_name": doc["policy_name"],
                        "text": " ".join(current_chunk),
                        "chunk_index": chunk_index,
                    })
                    current_chunk = []
                    current_len = 0
                    chunk_index += 1
            if current_chunk:
                chunks.append({
                    "id": f"{doc['id']}_chunk_{chunk_index}",
                    "doc_id": doc["id"],
                    "policy_name": doc["policy_name"],
                    "text": " ".join(current_chunk),
                    "chunk_index": chunk_index,
                })
    print(f"  Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks


def embed_and_store(chunks: list[dict], chroma_dir: str, collection_name: str):
    """Embed all chunks with all-MiniLM-L6-v2 and store in ChromaDB."""
    print(f"  Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=chroma_dir)

    # Delete and recreate collection for idempotent runs
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection '{collection_name}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert into ChromaDB
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "doc_id": c["doc_id"],
                "policy_name": c["policy_name"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )
    print(f"  [OK] Stored {len(chunks)} chunks in ChromaDB collection '{collection_name}'.")
    print(f"  ChromaDB location: {chroma_dir}")
    return collection


def run_ingest():
    print("=" * 60)
    print("MODULE 3: Ingestion Pipeline")
    print("=" * 60)

    documents = load_documents(DOCS_DIR)
    chunks = chunk_documents(documents)
    collection = embed_and_store(chunks, CHROMA_DIR, COLLECTION_NAME)

    # Quick verification query
    print("\nVerification [*] test retrieval for 'delivery fee':")
    model = SentenceTransformer(EMBEDDING_MODEL)
    test_query = "What is the delivery fee?"
    test_embedding = model.encode([test_query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=test_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"  [{i+1}] doc_id={meta['doc_id']}, policy={meta['policy_name']}, distance={dist:.4f}")
        print(f"       Snippet: {doc[:100]}...")

    print("\n[*] Ingestion complete!")


if __name__ == "__main__":
    run_ingest()
