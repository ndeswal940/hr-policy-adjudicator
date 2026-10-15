import os
import chromadb

def build_vector_store():
    # Persistent ChromaDB client in ./chroma_db
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="hr_policies")

    policy_file_path = os.path.join("data", "hr_policies.txt")
    if not os.path.exists(policy_file_path):
        raise FileNotFoundError(f"Missing policy file: {policy_file_path}")

    with open(policy_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split into discrete clause blocks
    chunks = [c.strip() for c in raw_text.split("\n\n") if c.strip()]
    chunk_ids = [f"policy_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=chunk_ids,
        metadatas=[{"source": "HR_Policy_Handbook"} for _ in chunks]
    )
    print(f"Successfully indexed {len(chunks)} policy clauses into ChromaDB.")

if __name__ == "__main__":
    build_vector_store()
