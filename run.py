from doc_store import get_docs_store, get_embeddings
from llm_pipeline import ask_llm, get_llm_pipeline
from file_loader import load_file

MODEL="llama3.1"
# MODEL="phi3:medium"
# MODEL="gemma2:9b"
# MODEL="mistral"
PROMPT="prompt6.txt"

TICKET="ticket2.txt"

query = load_file("input", TICKET)

document_store = get_docs_store(MODEL)
query_embedding = get_embeddings(MODEL, query)

template = load_file("templates", PROMPT)

pipe = get_llm_pipeline(MODEL, document_store, template)

response = ask_llm(pipe, query, query_embedding)
print(response)
