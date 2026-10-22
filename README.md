
# 📘 Personal Knowledge Base — Semantic Search with ChromaDB

A persistent, metadata‑rich knowledge base built using **ChromaDB** and **SentenceTransformer embeddings**.  
This project demonstrates how to store course concepts, embed them semantically, and retrieve them using both broad and filtered queries.

---

## 🚀 Features

- **Persistent ChromaDB storage**  
  Your knowledge base survives script restarts and grows over time.

- **Semantic embeddings (all‑MiniLM‑L6‑v2)**  
  Queries match meaning, not keywords.

- **Metadata‑driven filtering**  
  Search by module, topic, or both.

- **15 curated documents**  
  Covering FastAPI, authentication, databases, embeddings, semantic search, chunking, and Streamlit.

- **Three search modes**
  - Broad semantic search  
  - Filtered search (module‑specific)  
  - Semantic mismatch search (different wording, same meaning)

---

## 📂 Project Structure

```
knowledge_base.py
my_knowledge_DB/     # Persistent ChromaDB storage
README.md
```

---

## 📦 Installation

Create a virtual environment and install dependencies:

```bash
pip install chromadb sentence-transformers
```

---

## 🧠 How It Works

### 1. Create a persistent ChromaDB client

```python
client = chromadb.PersistentClient(path="./my_knowledge_DB")
```

### 2. Attach an embedding function

```python
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```

### 3. Create or load a collection

```python
collection = client.get_or_create_collection(
    name="my_knowledge",
    embedding_function=embedder,
    metadata={"description": "AI Engineering Foundations course content"},
    configuration={"hnsw": {"space": "cosine"}}
)
```

### 4. Upsert documents with metadata

Each document includes:

- `id`
- `text`
- `metadata` → `{module, topic}`

### 5. Run semantic queries

- Broad query  
- Filtered query  
- Semantic mismatch query

---

## 🔍 Example Queries

### **Broad Query**
Searches the entire knowledge base:

```python
collection.query(
    query_texts=["How does semantic search work?"],
    n_results=5
)
```

### **Filtered Query**
Restricts results to module 5:

```python
collection.query(
    query_texts=["How does semantic search work?"],
    where={"module": "5"},
    n_results=5
)
```

### **Semantic Mismatch Query**
Uses different wording to test semantic understanding:

```python
collection.query(
    query_texts=["How can a system understand meaning instead of exact words?"],
    n_results=5
)
```

---

## 📊 Sample Output (Structure)

Each query returns:

- `documents`
- `metadatas`
- `distances`

Example:

```
[0.12] Embeddings convert text into numerical vectors... | {'module': '7', 'topic': 'ai'}
```

---

## 🎯 What This Project Demonstrates

- How embeddings enable semantic search  
- How metadata improves filtering and organization  
- How ChromaDB can serve as a long‑term knowledge store  
- How different queries produce different retrieval patterns  

This project mirrors real‑world retrieval workflows used in AI assistants, RAG systems, and intelligent search tools.

---

## 📈 Future Enhancements

- Add more modules and topics  
- Build a Streamlit UI for interactive querying  
- Add similarity thresholding  
- Visualize embeddings with PCA or UMAP  

