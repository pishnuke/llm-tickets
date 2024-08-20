from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from haystack_integrations.components.generators.ollama import OllamaGenerator

LLM_URL="http://localhost:11434/api/generate"

def get_llm_pipeline(model: str, document_store, template: str):
  params = { "temperature": 0 }
  pipe = Pipeline()

  pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=10))
  pipe.add_component("prompt_builder", PromptBuilder(template=template))
  pipe.add_component("llm", OllamaGenerator(model=model, url=LLM_URL, generation_kwargs=params))

  pipe.connect("retriever", "prompt_builder.documents")
  pipe.connect("prompt_builder", "llm")
  return pipe


def ask_llm(pipe, query: str, query_embedding: list) -> str:
  response = pipe.run({"prompt_builder": {"query": query}, "retriever": {"query_embedding": query_embedding}})

  return "".join(response["llm"]["replies"])
