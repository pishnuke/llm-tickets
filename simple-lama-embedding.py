import os
import glob

from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder


MODEL="llama3.1"
PROMPT="prompt2.txt"
TICKET="ticket2.txt"

document_store = InMemoryDocumentStore(embedding_similarity_function="dot_product")
doc_embedder = OllamaDocumentEmbedder(model=MODEL, url="http://localhost:11434/api/embeddings")
text_embedder = OllamaTextEmbedder(model=MODEL, url="http://localhost:11434/api/embeddings")

doc_paths = glob.glob(os.path.join("docs", '*.txt'))
docs = [Document(content=open(file_path, 'r').read()) for file_path in doc_paths]
document_store.write_documents(doc_embedder.run(docs)["documents"])

template = open(os.path.join("templates", PROMPT), 'r').read()

pipe = Pipeline()

pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=3))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", OllamaGenerator(model=MODEL, url="http://localhost:11434/api/generate"))

pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

query = open(TICKET, 'r').read()
query_embedding = text_embedder.run(query)["embedding"]

response = pipe.run({"prompt_builder": {"query": query}, "retriever": {"query_embedding": query_embedding}})

print("".join(response["llm"]["replies"]))
