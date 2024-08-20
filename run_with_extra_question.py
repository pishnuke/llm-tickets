from doc_store import get_docs_store, get_embeddings
from llm_pipeline import ask_llm, get_llm_pipeline
from file_loader import load_file

MODEL="llama3.1"
# MODEL="phi3:medium"
# MODEL="gemma2:9b"
# MODEL="mistral"
PROMPT="prompt6.txt"

TICKET="ticket1.txt"

query = load_file("input", TICKET)

document_store = get_docs_store(MODEL)
query_embedding = get_embeddings(MODEL, query)

template = load_file("templates", PROMPT)

pipe = get_llm_pipeline(MODEL, document_store, template)

response1_str = ask_llm(pipe, query, query_embedding)

followup_query = f"""
Question: {query}
Answer: {response1_str}

Follow-up Question: could your response be improved in any way? If so, rewrite it to be better. If not, just respond with <COMPLETE>
"""
response2_str = ask_llm(pipe, followup_query, query_embedding)

print(response1_str, "\n>>>\n", response2_str)
