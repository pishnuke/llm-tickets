from haystack.components.evaluators import SASEvaluator # https://docs.haystack.deepset.ai/docs/sasevaluator

from doc_store import get_docs_store, get_embeddings
from llm_pipeline import get_llm_pipeline
from file_loader import load_file, load_files

MODEL="llama3.1"
# MODEL="phi3:medium"
# MODEL="gemma2:9b"
# MODEL="mistral"
PROMPT="prompt3.txt"

tickets = load_files("input", "*.txt")
ground_truth = load_files("input", "*.validation")
assert len(tickets) == len(ground_truth)
predicted = []

document_store = get_docs_store(MODEL)
for ticket in tickets:
  print("Running:", ticket)
  query = ticket

  query_embedding = get_embeddings(MODEL, query)

  template = load_file("templates", PROMPT)

  pipe = get_llm_pipeline(MODEL, document_store, template)

  response = pipe.run({"prompt_builder": {"query": query}, "retriever": {"query_embedding": query_embedding}})

  predicted.append("".join(response["llm"]["replies"]))

sas_evaluator = SASEvaluator()
sas_evaluator.warm_up()
result = sas_evaluator.run(
  ground_truth_answers=ground_truth,
  predicted_answers=predicted
)
print(f"Model={MODEL} Prompt={PROMPT}")
print(f"Score={result['score']}")
print(f"Individual scores:\n{'\n'.join(map(str, zip(tickets, result['individual_scores'])))}")
