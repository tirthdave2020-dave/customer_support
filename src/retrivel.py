import json
from sentence_transformers import SentenceTransformer
import numpy as np


def load_cases(path="amazon_cases.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_documents(cases):
    documents = []

    for case in cases:
        customer_text = " ".join(case["customer_messages"])
        amazon_text = " ".join(case["amazon_replies"])

        document = (
            f"Customer problem:\n{customer_text}\n\n"
            f"AmazonHelp response:\n{amazon_text}"
        )

        documents.append(document)

    return documents
def create_embeddings(documents):
    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = model.encode(
        documents,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True

    )

    return embeddings
import faiss


embeddings = np.load("artifacts/demo_embeddings.npy").astype("float32")

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)



def retrieve_cases(query, k=5):
    # Load embedding model
    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Convert customer query into vector
    query_vector = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    # Search FAISS
    scores, indices = index.search(query_vector, k)

    # Load original cases
    cases = load_cases("artifacts/demo_cases.json")

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "case": cases[idx]
        })

    return results
