from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder

from file_loader import load_files

EMBEDDER_URL = "http://localhost:11434/api/embeddings"


def get_docs_store(model: str) -> tuple():
  document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
  doc_embedder = OllamaDocumentEmbedder(model=model, url=EMBEDDER_URL)

  docs_contents = load_files("docs", "*.txt")
  docs = [Document(content=content) for content in docs_contents]
  document_store.write_documents(doc_embedder.run(docs)["documents"])

  return document_store


def get_embeddings(model: str, query: str) -> list[float]:
  text_embedder = OllamaTextEmbedder(model=model, url=EMBEDDER_URL)

  query_embedding = text_embedder.run(query)["embedding"]

  return query_embedding
