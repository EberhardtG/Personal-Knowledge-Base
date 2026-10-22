import chromadb
from chromadb.utils import embedding_functions

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# Create a persistent client (data survives script restarts)
client = chromadb.PersistentClient(path="./my_knowledge_DB")

# Create (or get) a collection
# If it already exists, get_or_create_collection returns the existing one
collection = client.get_or_create_collection(
    name="my_knowledge",
    embedding_function=embedder,
    metadata={"description": "AI Engineering Foundations course content"},
    configuration={"hnsw": {"space": "cosine"}}
)

print(f"Collection '{collection.name}' ready.")
print(f"Current document count: {collection.count()}")

documents = [
    {
        "id": "doc_1",
        "text": "FastAPI automatically validates request data using Pydantic models, ensuring clean and predictable API inputs.",
        "metadata": {"module": "5", "topic": "api"}
    },
    {
        "id": "doc_2",
        "text": "Dependency injection in FastAPI allows shared resources like database sessions to be managed safely and consistently.",
        "metadata": {"module": "5", "topic": "api"}
    },
    {
        "id": "doc_3",
        "text": "JWT authentication provides stateless login by issuing signed tokens that clients include in protected API requests.",
        "metadata": {"module": "5", "topic": "auth"}
    },
    {
        "id": "doc_4",
        "text": "Password hashing ensures that stored credentials remain secure even if the database is compromised.",
        "metadata": {"module": "5", "topic": "auth"}
    },
    {
        "id": "doc_5",
        "text": "SQLAlchemy's ORM maps Python classes to database tables, enabling expressive queries without raw SQL.",
        "metadata": {"module": "6", "topic": "database"}
    },
    {
        "id": "doc_6",
        "text": "Database migrations allow schema changes to be tracked and applied consistently across environments.",
        "metadata": {"module": "6", "topic": "database"}
    },
    {
        "id": "doc_7",
        "text": "Embeddings convert text into numerical vectors that capture semantic meaning rather than exact wording.",
        "metadata": {"module": "7", "topic": "ai"}
    },
    {
        "id": "doc_8",
        "text": "Cosine similarity measures how close two embedding vectors are, enabling semantic search and ranking.",
        "metadata": {"module": "7", "topic": "ai"}
    },
    {
        "id": "doc_9",
        "text": "Chunking long documents improves search quality by splitting text into smaller, more meaningful units.",
        "metadata": {"module": "7", "topic": "ai"}
    },
    {
        "id": "doc_10",
        "text": "Fixed-size chunking creates uniform slices of text, which can help when queries match specific phrases.",
        "metadata": {"module": "7", "topic": "ai"}
    },
    {
        "id": "doc_11",
        "text": "Paragraph-based chunking preserves semantic boundaries, often producing more accurate search results.",
        "metadata": {"module": "7", "topic": "ai"}
    },
    {
        "id": "doc_12",
        "text": "Streamlit reruns the script on every interaction, so session state is required to persist user data.",
        "metadata": {"module": "8", "topic": "frontend"}
    },
    {
        "id": "doc_13",
        "text": "Streamlit widgets like sliders, text inputs, and sidebars enable rapid development of interactive dashboards.",
        "metadata": {"module": "8", "topic": "frontend"}
    },
    {
        "id": "doc_14",
        "text": "Semantic search systems embed documents once and embed queries on the fly, comparing them using similarity scores.",
        "metadata": {"module": "7", "topic": "ai"}
    },
    {
        "id": "doc_15",
        "text": "API error handling ensures that clients receive clear messages and proper HTTP status codes when requests fail.",
        "metadata": {"module": "5", "topic": "api"}
    }
]

# Add to ChromaDB using .upsert
collection.upsert(
    ids=[d["id"] for d in documents],
    documents=[d["text"] for d in documents],
    metadatas=[d["metadata"] for d in documents]
)


print(f"Added {len(documents)} documents.")
print(f"Collection now has {collection.count()} documents.")

# Broad query
results_broad= collection.query(
    query_texts=["How does semantic search work?"],
    n_results=5
)

# Filtered query
results_filtered = collection.query(
    query_texts=["How does semantic search work?"],
    where={"module": "5"},
    n_results=5
)

# Semantic mismatch query
results_semantic = collection.query(
    query_texts=["How can a system understand meaning instead of exact words?"],
    n_results=5
)

print("\n=== Search Results ===")
print(f"Query: 'How does semantic search work?'\n")

print("\n--- Broad Query Results ---")
for doc, meta, dist in zip(
    results_broad["documents"][0],
    results_broad["metadatas"][0],
    results_broad["distances"][0]
):
    print(f"[{dist:.4f}] {doc}  |  {meta}")


print("\n---Filtered Query Results---")
for doc, meta, dist in zip(
    results_filtered["documents"][0],
    results_filtered["metadatas"][0],
    results_filtered["distances"][0]
):
    print(f"[{dist:.4f}] {doc} | {meta}")


print("\n---Semantic Query Results---")
for doc, meta, dist in zip(
    results_semantic["documents"][0],
    results_semantic["metadatas"][0],
    results_semantic["distances"][0]
):
    print(f"[{dist:.4f}] {doc} | {meta}")

